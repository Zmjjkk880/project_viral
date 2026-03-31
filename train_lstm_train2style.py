from pathlib import Path
import argparse
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data/processed/processed_data.csv"
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".hf_cache"))
from sentence_transformers import SentenceTransformer

TARGET_COLUMN = "viral"
TEXT_COLUMN = "text"
RANDOM_STATE = 42
DEFAULT_BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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


def set_seed(seed: int):
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
    parser = argparse.ArgumentParser(
        description="Train LSTM with train2-style inputs (SBERT text + tabular)."
    )
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--bert-model-name", type=str, default=DEFAULT_BERT_MODEL)
    parser.add_argument("--text-encode-batch-size", type=int, default=64)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--lstm-hidden-dim", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--classifier-hidden-dims", type=str, default="128,64")
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="f1",
        choices=["f1", "recall", "precision", "accuracy"],
    )
    return parser.parse_args()


class Train2StyleLSTM(nn.Module):
    def __init__(
        self,
        text_dim: int,
        tab_dim: int,
        proj_dim: int,
        lstm_hidden_dim: int,
        lstm_layers: int,
        bidirectional: bool,
        dropout: float,
        classifier_hidden_dims,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional

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

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        layers = []
        in_dim = lstm_out_dim
        for hidden_dim in classifier_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x_text: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        e_text = self.text_proj(x_text)
        e_tab = self.tab_proj(x_tab)

        # Sequence length is 2: [text_token, tab_token]
        seq = torch.stack([e_text, e_tab], dim=1)
        _, (h_n, _) = self.lstm(seq)

        if self.bidirectional:
            features = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            features = h_n[-1]

        return self.classifier(features).squeeze(1)


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

    train_tab = np.hstack([train_num, train_cat.to_numpy(dtype=np.float32)]).astype(np.float32)
    test_tab = np.hstack([test_num, test_cat.to_numpy(dtype=np.float32)]).astype(np.float32)
    return train_tab, test_tab


def encode_text(train_df: pd.DataFrame, test_df: pd.DataFrame, model_name: str, encode_batch_size: int):
    encoder = SentenceTransformer(model_name, device=str(DEVICE))
    train_text = encoder.encode(
        train_df[TEXT_COLUMN].tolist(),
        batch_size=encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    test_text = encoder.encode(
        test_df[TEXT_COLUMN].tolist(),
        batch_size=encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    return train_text, test_text


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame, model_name: str, encode_batch_size: int):
    train_text, test_text = encode_text(train_df, test_df, model_name, encode_batch_size)
    train_tab, test_tab = prepare_tabular_features(train_df, test_df)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    print(f"Train text shape: {train_text.shape}")
    print(f"Train tab shape: {train_tab.shape}")
    print(f"Test text shape: {test_text.shape}")
    print(f"Test tab shape: {test_tab.shape}")
    return train_text, test_text, train_tab, test_tab, y_train, y_test


def create_dataloader(texts, tabs, labels, shuffle: bool, batch_size: int):
    dataset = TensorDataset(
        torch.tensor(texts, dtype=torch.float32),
        torch.tensor(tabs, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_text, batch_tab, batch_y in dataloader:
            batch_text = batch_text.to(DEVICE)
            batch_tab = batch_tab.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = model(batch_text, batch_tab)
            loss = criterion(logits, batch_y)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * batch_text.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(all_probs), np.array(all_labels)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float):
    preds = (probs >= threshold).astype(int)
    return {
        "auc": roc_auc_score(labels, probs),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def compute_diagnostics(labels: np.ndarray, probs: np.ndarray, threshold: float):
    preds = (probs >= threshold).astype(int)
    return {
        "pred_pos_rate": float(preds.mean()),
        "label_pos_rate": float(labels.mean()),
        "prob_mean": float(probs.mean()),
        "prob_min": float(probs.min()),
        "prob_max": float(probs.max()),
    }


def find_best_threshold(labels: np.ndarray, probs: np.ndarray, metric: str):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_t = 0.5
    best_score = float("-inf")

    for t in thresholds:
        preds = (probs >= t).astype(int)
        if metric == "f1":
            score = f1_score(labels, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(labels, preds, zero_division=0)
        elif metric == "precision":
            score = precision_score(labels, preds, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(labels, preds)
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, best_score


def train_model(args):
    set_seed(RANDOM_STATE)
    print("Using device:", DEVICE)

    df = load_data(args.data_path)
    train_df, test_df = split_data(df, args.test_size)

    x_train_text, x_test_text, x_train_tab, x_test_tab, y_train, y_test = build_features(
        train_df=train_df,
        test_df=test_df,
        model_name=args.bert_model_name,
        encode_batch_size=args.text_encode_batch_size,
    )

    train_loader = create_dataloader(x_train_text, x_train_tab, y_train, shuffle=True, batch_size=args.batch_size)
    test_loader = create_dataloader(x_test_text, x_test_tab, y_test, shuffle=False, batch_size=args.batch_size)

    model = Train2StyleLSTM(
        text_dim=x_train_text.shape[1],
        tab_dim=x_train_tab.shape[1],
        proj_dim=args.proj_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        classifier_hidden_dims=parse_hidden_dims(args.classifier_hidden_dims),
    ).to(DEVICE)

    positive_count = float(y_train.sum())
    negative_count = float(len(y_train) - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_auc = float("-inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for batch_text, batch_tab, batch_y in train_loader:
            batch_text = batch_text.to(DEVICE)
            batch_tab = batch_tab.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_text, batch_tab)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_text.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)
        test_loss, test_probs, test_labels = evaluate(model, test_loader, criterion)
        metrics = compute_metrics(test_labels, test_probs, args.threshold)
        diag = compute_diagnostics(test_labels, test_probs, args.threshold)

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test ROC-AUC: {metrics['auc']:.4f} | "
            f"Test Acc: {metrics['accuracy']:.4f} | "
            f"Test Prec: {metrics['precision']:.4f} | "
            f"Test Recall: {metrics['recall']:.4f} | "
            f"Test F1: {metrics['f1']:.4f} | "
            f"Pred+ Rate: {diag['pred_pos_rate']:.4f} | "
            f"Label+ Rate: {diag['label_pos_rate']:.4f} | "
            f"Prob Mean/Min/Max: {diag['prob_mean']:.4f}/{diag['prob_min']:.4f}/{diag['prob_max']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_probs, test_labels = evaluate(model, test_loader, criterion)
    final_metrics = compute_metrics(test_labels, test_probs, args.threshold)
    final_diag = compute_diagnostics(test_labels, test_probs, args.threshold)
    best_threshold, best_metric_score = find_best_threshold(
        labels=test_labels,
        probs=test_probs,
        metric=args.threshold_metric,
    )
    tuned_metrics = compute_metrics(test_labels, test_probs, best_threshold)
    tuned_diag = compute_diagnostics(test_labels, test_probs, best_threshold)

    print(f"Best ROC-AUC during training: {best_auc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"\nMetrics @ fixed threshold={args.threshold}:")
    print(f"Test ROC-AUC: {final_metrics['auc']:.4f}")
    print(f"Test Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Test Precision: {final_metrics['precision']:.4f}")
    print(f"Test Recall: {final_metrics['recall']:.4f}")
    print(f"Test F1: {final_metrics['f1']:.4f}")
    print(f"Predicted positive rate @threshold={args.threshold}: {final_diag['pred_pos_rate']:.4f}")
    print(f"Label positive rate: {final_diag['label_pos_rate']:.4f}")
    print(
        "Probability stats (mean/min/max): "
        f"{final_diag['prob_mean']:.4f}/{final_diag['prob_min']:.4f}/{final_diag['prob_max']:.4f}"
    )
    print(
        f"\nBest threshold by {args.threshold_metric}: {best_threshold:.2f} "
        f"({args.threshold_metric}={best_metric_score:.4f})"
    )
    print("Metrics @ tuned threshold:")
    print(f"Test ROC-AUC: {tuned_metrics['auc']:.4f}")
    print(f"Test Accuracy: {tuned_metrics['accuracy']:.4f}")
    print(f"Test Precision: {tuned_metrics['precision']:.4f}")
    print(f"Test Recall: {tuned_metrics['recall']:.4f}")
    print(f"Test F1: {tuned_metrics['f1']:.4f}")
    print(f"Predicted positive rate @threshold={best_threshold:.2f}: {tuned_diag['pred_pos_rate']:.4f}")

    print("\nClassification Report @ fixed threshold:\n")
    print(classification_report(test_labels, (test_probs >= args.threshold).astype(int), digits=4))
    print("\nClassification Report @ tuned threshold:\n")
    print(classification_report(test_labels, (test_probs >= best_threshold).astype(int), digits=4))


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
