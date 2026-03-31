from pathlib import Path
import argparse
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data/processed/processed_data.csv"
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".hf_cache"))
from transformers import AutoTokenizer

TARGET_COLUMN = "viral"
TEXT_COLUMN = "text"
RANDOM_STATE = 42

NUMERIC_COLUMNS = [
    "upload_hour",
    "is_weekend",
    "duration_sec",
    "creator_avg_views",
    "title_length",
    "has_emoji",
]

CATEGORICAL_COLUMNS = [
    "category",
    "genre",
    "sound_type",
    "music_track",
    "publish_dayofweek",
    "publish_period",
    "season",
    "event_season",
    "creator_tier",
    "platform",
    "country",
    "region",
    "language",
    "traffic_source",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_hidden_dims(raw: str):
    if not raw.strip():
        return []
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Train BiLSTM + tabular model for viral video prediction.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--tokenizer-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--lstm-hidden-dim", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument(
        "--token-mixer",
        type=str,
        default="projected_concat",
        choices=["raw_concat", "projected_concat", "weighted_sum", "attention_pool"],
    )
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument("--classifier-hidden-dims", type=str, default="128")
    return parser.parse_args()


class ViralDataset(Dataset):
    def __init__(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        num_features: np.ndarray,
        cat_features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.num_features = torch.tensor(num_features, dtype=torch.float32)
        self.cat_features = torch.tensor(cat_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.num_features[idx],
            self.cat_features[idx],
            self.labels[idx],
        )


class TokenMixer(nn.Module):
    def __init__(
        self,
        text_dim: int,
        num_dim: int,
        cat_dim: int,
        token_mixer: str,
        proj_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.num_dim = num_dim
        self.cat_dim = cat_dim
        self.token_mixer = token_mixer

        if token_mixer == "raw_concat":
            self.output_dim = text_dim + num_dim + cat_dim
        elif token_mixer == "projected_concat":
            self.text_proj = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
            self.num_proj = nn.Sequential(nn.Linear(num_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
            self.cat_proj = nn.Sequential(nn.Linear(cat_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
            self.output_dim = proj_dim * 3
        elif token_mixer == "weighted_sum":
            self.text_proj = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
            self.num_proj = nn.Sequential(nn.Linear(num_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
            self.cat_proj = nn.Sequential(nn.Linear(cat_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
            self.text_gate = nn.Linear(proj_dim, 1)
            self.num_gate = nn.Linear(proj_dim, 1)
            self.cat_gate = nn.Linear(proj_dim, 1)
            self.output_dim = proj_dim
        elif token_mixer == "attention_pool":
            self.text_proj = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
            self.num_proj = nn.Sequential(nn.Linear(num_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
            self.cat_proj = nn.Sequential(nn.Linear(cat_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
            self.attention = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=1, batch_first=True, dropout=dropout)
            self.output_dim = proj_dim
        else:
            raise ValueError(f"Unsupported token_mixer: {token_mixer}")

    def forward(self, x_text: torch.Tensor, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if self.token_mixer == "raw_concat":
            return torch.cat([x_text, x_num, x_cat], dim=1)

        if self.token_mixer == "projected_concat":
            e_text = self.text_proj(x_text)
            e_num = self.num_proj(x_num)
            e_cat = self.cat_proj(x_cat)
            return torch.cat([e_text, e_num, e_cat], dim=1)

        if self.token_mixer == "weighted_sum":
            e_text = self.text_proj(x_text)
            e_num = self.num_proj(x_num)
            e_cat = self.cat_proj(x_cat)
            gates = torch.cat([self.text_gate(e_text), self.num_gate(e_num), self.cat_gate(e_cat)], dim=1)
            weights = torch.softmax(gates, dim=1)
            return (
                weights[:, 0:1] * e_text
                + weights[:, 1:2] * e_num
                + weights[:, 2:3] * e_cat
            )

        if self.token_mixer == "attention_pool":
            e_text = self.text_proj(x_text)
            e_num = self.num_proj(x_num)
            e_cat = self.cat_proj(x_cat)
            tokens = torch.stack([e_text, e_num, e_cat], dim=1)
            attn_out, _ = self.attention(tokens, tokens, tokens)
            return attn_out.mean(dim=1)

        raise ValueError(f"Unsupported token_mixer: {self.token_mixer}")


class BiLSTMTabularClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        embed_dim: int,
        lstm_hidden_dim: int,
        lstm_layers: int,
        bidirectional: bool,
        num_dim: int,
        cat_dim: int,
        token_mixer: str,
        proj_dim: int,
        dropout: float,
        classifier_hidden_dims,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm_layers = lstm_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        text_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        self.token_mixer = TokenMixer(
            text_dim=text_dim,
            num_dim=num_dim,
            cat_dim=cat_dim,
            token_mixer=token_mixer,
            proj_dim=proj_dim,
            dropout=dropout,
        )

        layers = []
        in_dim = self.token_mixer.output_dim
        for hidden_dim in classifier_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).cpu()
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)

        if self.bidirectional:
            # Last layer forward/backward hidden states.
            forward_hidden = h_n[-2]
            backward_hidden = h_n[-1]
            return torch.cat([forward_hidden, backward_hidden], dim=1)

        return h_n[-1]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> torch.Tensor:
        text_repr = self._encode_text(input_ids, attention_mask)
        mixed = self.token_mixer(text_repr, x_num, x_cat)
        return self.classifier(mixed).squeeze(1)


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame, test_size: float):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    return train_df, test_df


def prepare_tabular_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_df[NUMERIC_COLUMNS]).astype(np.float32)
    test_num = scaler.transform(test_df[NUMERIC_COLUMNS]).astype(np.float32)

    train_cat = pd.get_dummies(train_df[CATEGORICAL_COLUMNS], drop_first=False)
    test_cat = pd.get_dummies(test_df[CATEGORICAL_COLUMNS], drop_first=False)
    test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)

    train_cat = train_cat.to_numpy(dtype=np.float32)
    test_cat = test_cat.to_numpy(dtype=np.float32)
    return train_num, test_num, train_cat, test_cat


def tokenize_text(tokenizer, train_texts, test_texts, max_length: int):
    train_enc = tokenizer(
        train_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    test_enc = tokenizer(
        test_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return train_enc, test_enc


def build_dataloaders(train_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer, max_length: int, batch_size: int):
    train_num, test_num, train_cat, test_cat = prepare_tabular_features(train_df, test_df)
    train_enc, test_enc = tokenize_text(
        tokenizer,
        train_df[TEXT_COLUMN].tolist(),
        test_df[TEXT_COLUMN].tolist(),
        max_length=max_length,
    )

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    train_dataset = ViralDataset(
        input_ids=train_enc["input_ids"],
        attention_mask=train_enc["attention_mask"],
        num_features=train_num,
        cat_features=train_cat,
        labels=y_train,
    )
    test_dataset = ViralDataset(
        input_ids=test_enc["input_ids"],
        attention_mask=test_enc["attention_mask"],
        num_features=test_num,
        cat_features=test_cat,
        labels=y_test,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, y_train, y_test, train_num.shape[1], train_cat.shape[1]


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, x_num, x_cat, labels in dataloader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            x_num = x_num.to(DEVICE)
            x_cat = x_cat.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(input_ids, attention_mask, x_num, x_cat)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * input_ids.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(all_probs), np.array(all_labels)


def train_model(args):
    set_seed(RANDOM_STATE)
    print("Using device:", DEVICE)

    df = load_data(args.data_path)
    train_df, test_df = split_data(df, args.test_size)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    train_loader, test_loader, y_train, _, num_dim, cat_dim = build_dataloaders(
        train_df=train_df,
        test_df=test_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    classifier_hidden_dims = parse_hidden_dims(args.classifier_hidden_dims)
    model = BiLSTMTabularClassifier(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        embed_dim=args.embed_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        num_dim=num_dim,
        cat_dim=cat_dim,
        token_mixer=args.token_mixer,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        classifier_hidden_dims=classifier_hidden_dims,
    ).to(DEVICE)

    positive_count = float(y_train.sum())
    negative_count = float(len(y_train) - positive_count)
    pos_ratio = negative_count / max(positive_count, 1.0)
    pos_weight = torch.tensor([pos_ratio], dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_auc = float("-inf")
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for input_ids, attention_mask, x_num, x_cat, labels in train_loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            x_num = x_num.to(DEVICE)
            x_cat = x_cat.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, x_num, x_cat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * input_ids.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        test_loss, test_probs_epoch, test_labels_epoch = evaluate(model, test_loader, criterion)
        test_auc = roc_auc_score(test_labels_epoch, test_probs_epoch)

        if test_auc > best_auc:
            best_auc = test_auc
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test ROC-AUC: {test_auc:.4f}"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_probs, test_labels = evaluate(model, test_loader, criterion)

    print(f"Best ROC-AUC during training: {best_auc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(test_labels, test_probs):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(test_labels, (test_probs >= args.threshold).astype(int), digits=4))


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
