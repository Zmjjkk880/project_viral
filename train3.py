from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data/processed/processed_data.csv"

TARGET_COLUMN = "viral"
TEXT_COLUMN = "text"
RANDOM_STATE = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_HIDDEN_DIM = 128
DEFAULT_DROPOUT = 0.3
DEFAULT_TEST_SIZE = 0.2
DEFAULT_THRESHOLD = 0.5
BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Use CUDA on Colab if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP for viral video prediction.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument(
        "--token-mixer",
        type=str,
        default="raw_concat",
        choices=["raw_concat", "projected_concat", "weighted_sum", "attention_pool"],
    )
    parser.add_argument("--proj-dim", type=int, default=64)
    return parser.parse_args()




# TokenMixer module for fusion logic (3 groups: text + numeric + categorical)
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
        self.proj_dim = proj_dim

        if token_mixer == "raw_concat":
            self.output_dim = text_dim + num_dim + cat_dim
        elif token_mixer == "projected_concat":
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.num_proj = nn.Sequential(
                nn.Linear(num_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.cat_proj = nn.Sequential(
                nn.Linear(cat_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.output_dim = proj_dim * 3
        elif token_mixer == "weighted_sum":
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.num_proj = nn.Sequential(
                nn.Linear(num_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.cat_proj = nn.Sequential(
                nn.Linear(cat_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.text_gate = nn.Linear(proj_dim, 1)
            self.num_gate = nn.Linear(proj_dim, 1)
            self.cat_gate = nn.Linear(proj_dim, 1)
            self.output_dim = proj_dim
        elif token_mixer == "attention_pool":
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.num_proj = nn.Sequential(
                nn.Linear(num_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.cat_proj = nn.Sequential(
                nn.Linear(cat_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.attention = nn.MultiheadAttention(
                embed_dim=proj_dim,
                num_heads=1,
                batch_first=True,
                dropout=dropout,
            )
            self.output_dim = proj_dim
        else:
            raise ValueError(f"Unsupported token_mixer: {token_mixer}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_text = x[:, : self.text_dim]
        x_num = x[:, self.text_dim : self.text_dim + self.num_dim]
        x_cat = x[:, self.text_dim + self.num_dim :]

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
            gates = torch.cat(
                [self.text_gate(e_text), self.num_gate(e_num), self.cat_gate(e_cat)], dim=1
            )
            weights = torch.softmax(gates, dim=1)
            w_text = weights[:, 0].unsqueeze(1)
            w_num = weights[:, 1].unsqueeze(1)
            w_cat = weights[:, 2].unsqueeze(1)
            return w_text * e_text + w_num * e_num + w_cat * e_cat

        if self.token_mixer == "attention_pool":
            e_text = self.text_proj(x_text)
            e_num = self.num_proj(x_num)
            e_cat = self.cat_proj(x_cat)
            tokens = torch.stack([e_text, e_num, e_cat], dim=1)
            attn_out, _ = self.attention(tokens, tokens, tokens)
            return attn_out.mean(dim=1)

        raise ValueError(f"Unsupported token_mixer: {self.token_mixer}")


# Classifier using TokenMixer for fusion (3 groups: text + numeric + categorical)
class MLPClassifier(nn.Module):
    def __init__(
        self,
        text_dim: int,
        num_dim: int,
        cat_dim: int,
        hidden_dim: int,
        dropout: float,
        token_mixer: str,
        proj_dim: int,
    ) -> None:
        super().__init__()
        self.token_mixer = TokenMixer(
            text_dim=text_dim,
            num_dim=num_dim,
            cat_dim=cat_dim,
            token_mixer=token_mixer,
            proj_dim=proj_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.token_mixer.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.token_mixer(x)
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



def encode_text(train_df: pd.DataFrame, test_df: pd.DataFrame):
    encoder = SentenceTransformer(BERT_MODEL_NAME, device=str(DEVICE))

    train_text = encoder.encode(
        train_df[TEXT_COLUMN].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    test_text = encoder.encode(
        test_df[TEXT_COLUMN].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    return train_text, test_text



def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_text, test_text = encode_text(train_df, test_df)
    train_num, test_num, train_cat, test_cat = prepare_tabular_features(train_df, test_df)

    x_train = np.hstack([train_text, train_num, train_cat]).astype(np.float32)
    x_test = np.hstack([test_text, test_num, test_cat]).astype(np.float32)

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    print(f"Train text shape: {train_text.shape}")
    print(f"Train numeric shape: {train_num.shape}")
    print(f"Train categorical shape: {train_cat.shape}")
    print(f"Test text shape: {test_text.shape}")
    print(f"Test numeric shape: {test_num.shape}")
    print(f"Test categorical shape: {test_cat.shape}")
    print(f"Train feature shape: {x_train.shape}")
    print(f"Test feature shape: {x_test.shape}")

    return x_train, x_test, y_train, y_test, train_num.shape[1], train_cat.shape[1]



def create_dataloader(features: np.ndarray, labels: np.ndarray, shuffle: bool, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * batch_x.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(all_probs), np.array(all_labels)



def train_model(args):
    set_seed(RANDOM_STATE)

    df = load_data(args.data_path)
    train_df, test_df = split_data(df, args.test_size)

    x_train, x_test, y_train, y_test, num_dim, cat_dim = build_features(train_df, test_df)

    train_loader = create_dataloader(x_train, y_train, shuffle=True, batch_size=args.batch_size)
    test_loader = create_dataloader(x_test, y_test, shuffle=False, batch_size=args.batch_size)

    text_dim = x_train.shape[1] - num_dim - cat_dim
    model = MLPClassifier(
        text_dim=text_dim,
        num_dim=num_dim,
        cat_dim=cat_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        token_mixer=args.token_mixer,
        proj_dim=args.proj_dim,
    ).to(DEVICE)

    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_auc = float("-inf")
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_x.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        test_loss, test_probs_epoch, test_labels_epoch = evaluate(model, test_loader, criterion)
        test_auc = roc_auc_score(test_labels_epoch, test_probs_epoch)

        if test_auc > best_auc:
            best_auc = test_auc
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

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

    print(f"\nBatch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Dropout: {args.dropout}")
    print(f"Test size: {args.test_size}")
    print(f"Prediction threshold: {args.threshold}")
    print(f"Data path: {args.data_path}")
    print(f"Token mixer: {args.token_mixer}")
    print(f"Projection dim: {args.proj_dim}")

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(test_labels, test_probs):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(test_labels, (test_probs >= args.threshold).astype(int), digits=4))


if __name__ == "__main__":
    args = parse_args()
    train_model(args)