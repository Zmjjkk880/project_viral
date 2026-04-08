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
        choices=["text_only", "tabular_only", "raw_concat", "projected_concat", "weighted_sum", "attention_pool"],
    )
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(BASE_DIR / "comparison_outputs" / "model_probabilities.csv"),
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Train and compare all token mixers. If not set, only the token mixer specified by --token-mixer is trained.",
    )
    return parser.parse_args()




# TokenMixer module for fusion logic
class TokenMixer(nn.Module):
    def __init__(
        self,
        text_dim: int,
        tab_dim: int,
        token_mixer: str,
        proj_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.tab_dim = tab_dim
        self.token_mixer = token_mixer
        self.proj_dim = proj_dim

        if token_mixer == "text_only":
            self.output_dim = text_dim
        elif token_mixer == "tabular_only":
            self.output_dim = tab_dim
        elif token_mixer == "raw_concat":
            self.output_dim = text_dim + tab_dim
        elif token_mixer == "projected_concat":
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.tab_proj = nn.Sequential(
                nn.Linear(tab_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.output_dim = proj_dim * 2
        elif token_mixer == "weighted_sum":
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.tab_proj = nn.Sequential(
                nn.Linear(tab_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.text_gate = nn.Linear(proj_dim, 1)
            self.tab_gate = nn.Linear(proj_dim, 1)
            self.output_dim = proj_dim
        elif token_mixer == "attention_pool":
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.tab_proj = nn.Sequential(
                nn.Linear(tab_dim, proj_dim),
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
        x_tab = x[:, self.text_dim :]

        if self.token_mixer == "text_only":
            return x_text

        if self.token_mixer == "tabular_only":
            return x_tab

        if self.token_mixer == "raw_concat":
            return torch.cat([x_text, x_tab], dim=1)

        if self.token_mixer == "projected_concat":
            e_text = self.text_proj(x_text)
            e_tab = self.tab_proj(x_tab)
            return torch.cat([e_text, e_tab], dim=1)

        if self.token_mixer == "weighted_sum":
            e_text = self.text_proj(x_text)
            e_tab = self.tab_proj(x_tab)
            gates = torch.cat([self.text_gate(e_text), self.tab_gate(e_tab)], dim=1)
            weights = torch.softmax(gates, dim=1)
            w_text = weights[:, 0].unsqueeze(1)
            w_tab = weights[:, 1].unsqueeze(1)
            return w_text * e_text + w_tab * e_tab

        if self.token_mixer == "attention_pool":
            e_text = self.text_proj(x_text)
            e_tab = self.tab_proj(x_tab)
            tokens = torch.stack([e_text, e_tab], dim=1)
            attn_out, _ = self.attention(tokens, tokens, tokens)
            return attn_out.mean(dim=1)

        raise ValueError(f"Unsupported token_mixer: {self.token_mixer}")

    def get_weighted_sum_weights(self, x: torch.Tensor):
        if self.token_mixer != "weighted_sum":
            return None

        x_text = x[:, : self.text_dim]
        x_tab = x[:, self.text_dim :]
        e_text = self.text_proj(x_text)
        e_tab = self.tab_proj(x_tab)
        gates = torch.cat([self.text_gate(e_text), self.tab_gate(e_tab)], dim=1)
        weights = torch.softmax(gates, dim=1)
        return weights


# Classifier using TokenMixer for fusion
class MLPClassifier(nn.Module):
    def __init__(
        self,
        text_dim: int,
        tab_dim: int,
        hidden_dim: int,
        dropout: float,
        token_mixer: str,
        proj_dim: int,
    ) -> None:
        super().__init__()
        self.token_mixer = TokenMixer(
            text_dim=text_dim,
            tab_dim=tab_dim,
            token_mixer=token_mixer,
            proj_dim=proj_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.token_mixer.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
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

    train_num = scaler.fit_transform(train_df[NUMERIC_COLUMNS])
    test_num = scaler.transform(test_df[NUMERIC_COLUMNS])

    train_cat = pd.get_dummies(train_df[CATEGORICAL_COLUMNS], drop_first=False)
    test_cat = pd.get_dummies(test_df[CATEGORICAL_COLUMNS], drop_first=False)

    test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)

    train_tab = np.hstack([train_num, train_cat.to_numpy(dtype=np.float32)])
    test_tab = np.hstack([test_num, test_cat.to_numpy(dtype=np.float32)])

    return train_tab, test_tab



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
    train_tab, test_tab = prepare_tabular_features(train_df, test_df)

    x_train = np.hstack([train_text, train_tab]).astype(np.float32)
    x_test = np.hstack([test_text, test_tab]).astype(np.float32)

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    print(f"Train feature shape: {x_train.shape}")
    print(f"Test feature shape: {x_test.shape}")

    return x_train, x_test, y_train, y_test



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
    all_weights = []

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

            mixer_weights = model.token_mixer.get_weighted_sum_weights(batch_x)
            if mixer_weights is not None:
                all_weights.append(mixer_weights.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    weight_array = None
    if all_weights:
        weight_array = np.vstack(all_weights)

    return avg_loss, np.array(all_probs), np.array(all_labels), weight_array




def train_single_model(
    token_mixer: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    text_dim: int,
    tab_dim: int,
    args,
):
    train_loader = create_dataloader(x_train, y_train, shuffle=True, batch_size=args.batch_size)
    test_loader = create_dataloader(x_test, y_test, shuffle=False, batch_size=args.batch_size)

    model = MLPClassifier(
        text_dim=text_dim,
        tab_dim=tab_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        token_mixer=token_mixer,
        proj_dim=args.proj_dim,
    ).to(DEVICE)

    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_auc = float("-inf")
    best_model_state = None

    print(f"\n===== Training token mixer: {token_mixer} =====")

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
        test_loss, test_probs_epoch, test_labels_epoch, _ = evaluate(model, test_loader, criterion)
        test_auc = roc_auc_score(test_labels_epoch, test_probs_epoch)

        if test_auc > best_auc:
            best_auc = test_auc
            best_model_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test ROC-AUC: {test_auc:.4f}"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_probs, test_labels, test_weights = evaluate(model, test_loader, criterion)
    test_preds = (test_probs >= args.threshold).astype(int)
    test_auc = roc_auc_score(test_labels, test_probs)

    print(f"Best ROC-AUC during training ({token_mixer}): {best_auc:.4f}")
    print(f"Final Test Loss ({token_mixer}): {test_loss:.4f}")
    print(f"Final Test ROC-AUC ({token_mixer}): {test_auc:.4f}")
    if test_weights is not None:
        avg_text_weight = test_weights[:, 0].mean()
        avg_tab_weight = test_weights[:, 1].mean()
        print(
            f"Average weighted_sum weights ({token_mixer}): "
            f"text={avg_text_weight:.6f}, tabular={avg_tab_weight:.6f}"
        )

    return {
        "token_mixer": token_mixer,
        "test_loss": test_loss,
        "test_auc": test_auc,
        "probs": test_probs,
        "preds": test_preds,
        "labels": test_labels,
        "weights": test_weights,
    }


def train_model(args):
    set_seed(RANDOM_STATE)

    df = load_data(args.data_path)
    train_df, test_df = split_data(df, args.test_size)
    x_train, x_test, y_train, y_test = build_features(train_df, test_df)

    text_dim = x_train.shape[1] - len(NUMERIC_COLUMNS) - len(
        pd.get_dummies(train_df[CATEGORICAL_COLUMNS], drop_first=False).columns
    )
    tab_dim = x_train.shape[1] - text_dim

    all_token_mixers = ["text_only", "tabular_only", "raw_concat", "projected_concat", "weighted_sum", "attention_pool"]
    token_mixers = all_token_mixers if args.compare_all else [args.token_mixer]
    results = {}

    for token_mixer in token_mixers:
        set_seed(RANDOM_STATE)
        results[token_mixer] = train_single_model(
            token_mixer=token_mixer,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            text_dim=text_dim,
            tab_dim=tab_dim,
            args=args,
        )

    if args.compare_all:
        comparison_df = pd.DataFrame(
            {
                "true_label": results["text_only"]["labels"].astype(int),
                "text_only_prob": results["text_only"]["probs"],
                "tabular_only_prob": results["tabular_only"]["probs"],
                "raw_concat_prob": results["raw_concat"]["probs"],
                "projected_concat_prob": results["projected_concat"]["probs"],
                "weighted_sum_prob": results["weighted_sum"]["probs"],
                "attention_pool_prob": results["attention_pool"]["probs"],
            }
        )

        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path, index=False)

        print("\n===== Saved probability comparison CSV =====")
        print(f"Output path: {output_path}")

        print("\n===== Summary Metrics =====")
        for token_mixer in token_mixers:
            print(
                f"{token_mixer}: "
                f"loss={results[token_mixer]['test_loss']:.4f}, "
                f"auc={results[token_mixer]['test_auc']:.4f}"
            )

        print("\n===== Pairwise Prediction Differences =====")
        pairs = [
            ("text_only", "tabular_only"),
            ("text_only", "raw_concat"),
            ("text_only", "projected_concat"),
            ("text_only", "weighted_sum"),
            ("text_only", "attention_pool"),
            ("tabular_only", "raw_concat"),
            ("tabular_only", "projected_concat"),
            ("tabular_only", "weighted_sum"),
            ("tabular_only", "attention_pool"),
            ("raw_concat", "projected_concat"),
            ("raw_concat", "weighted_sum"),
            ("raw_concat", "attention_pool"),
            ("projected_concat", "weighted_sum"),
            ("projected_concat", "attention_pool"),
            ("weighted_sum", "attention_pool"),
        ]

        for left, right in pairs:
            pred_diff_count = (results[left]["preds"] != results[right]["preds"]).sum()
            mean_abs_prob_diff = np.abs(results[left]["probs"] - results[right]["probs"]).mean()
            max_abs_prob_diff = np.abs(results[left]["probs"] - results[right]["probs"]).max()
            print(
                f"{left} vs {right}: "
                f"different_preds={pred_diff_count}, "
                f"mean_abs_prob_diff={mean_abs_prob_diff:.6f}, "
                f"max_abs_prob_diff={max_abs_prob_diff:.6f}"
            )

        pred_matrix = np.column_stack([results[token_mixer]["preds"] for token_mixer in token_mixers])
        all_preds_same_count = np.all(pred_matrix == pred_matrix[:, [0]], axis=1).sum()
        print("\n===== Global Agreement =====")
        print(f"Samples with identical predicted labels across all 6 models: {all_preds_same_count}/{len(pred_matrix)}")

        print("\n===== Near-Threshold Counts =====")
        for token_mixer in token_mixers:
            probs = results[token_mixer]["probs"]
            near_threshold_count = ((probs > args.threshold - 0.05) & (probs < args.threshold + 0.05)).sum()
            print(
                f"{token_mixer}: "
                f"samples in [{args.threshold - 0.05:.2f}, {args.threshold + 0.05:.2f}] = {near_threshold_count}"
            )
    else:
        token_mixer = args.token_mixer
        print("\n===== Single Model Summary =====")
        print(f"Token mixer: {token_mixer}")
        print(f"Batch size: {args.batch_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Hidden dim: {args.hidden_dim}")
        print(f"Dropout: {args.dropout}")
        print(f"Test size: {args.test_size}")
        print(f"Prediction threshold: {args.threshold}")
        print(f"Data path: {args.data_path}")
        print(f"Projection dim: {args.proj_dim}")
        print(f"Test Loss: {results[token_mixer]['test_loss']:.4f}")
        print(f"Test ROC-AUC: {results[token_mixer]['test_auc']:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)