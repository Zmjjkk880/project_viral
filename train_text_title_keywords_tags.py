from pathlib import Path
import argparse
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data/processed/processed_data_20000_0.06.csv"

TARGET_COLUMN = "viral"
TEXT_COLUMN = "text"
RANDOM_STATE = 42

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_HIDDEN_DIM = 128
DEFAULT_DROPOUT = 0.3
DEFAULT_TEST_SIZE = 0.2
DEFAULT_THRESHOLD = 0.5
DEFAULT_BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train text-only binary classifier using selected segments from the structured text field."
    )
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--bert-model-name", type=str, default=DEFAULT_BERT_MODEL)
    parser.add_argument("--text-encode-batch-size", type=int, default=64)
    parser.add_argument("--use-title", action="store_true")
    parser.add_argument("--use-keywords", action="store_true")
    parser.add_argument("--use-tags", action="store_true")
    parser.add_argument("--use-hashtag", action="store_true")
    return parser.parse_args()


def extract_segment(series: pd.Series, start_tag: str, end_tag: str | None = None) -> pd.Series:
    pattern = re.escape(start_tag) + r"(.*?)"
    if end_tag is not None:
        pattern += re.escape(end_tag)

    extracted = series.fillna("").astype(str).str.extract(pattern, expand=False).fillna("")
    return extracted.str.strip()


def build_segment_text(df: pd.DataFrame, args) -> pd.DataFrame:
    base_text = df[TEXT_COLUMN].fillna("").astype(str)

    title = extract_segment(base_text, "[TITLE]", "[HASHTAG]")
    hashtag = extract_segment(base_text, "[HASHTAG]", "[KEYWORDS]")
    keywords = extract_segment(base_text, "[KEYWORDS]", "[TAGS]")
    tags = extract_segment(base_text, "[TAGS]")

    selected_segments = []
    selected_names = []

    if args.use_title:
        selected_segments.append(title)
        selected_names.append("title")
    if args.use_keywords:
        selected_segments.append(keywords)
        selected_names.append("keywords")
    if args.use_tags:
        selected_segments.append(tags)
        selected_names.append("tags")
    if args.use_hashtag:
        selected_segments.append(hashtag)
        selected_names.append("hashtag")

    if not selected_segments:
        selected_segments = [title, keywords, tags]
        selected_names = ["title", "keywords", "tags"]

    combined = selected_segments[0].copy()
    for part in selected_segments[1:]:
        combined = combined.str.cat(part, sep=" ")

    combined = combined.str.replace(r"\s+", " ", regex=True).str.strip()

    out_df = df.copy()
    out_df["title_text"] = title
    out_df["hashtag_text"] = hashtag
    out_df["keywords_text"] = keywords
    out_df["tags_text"] = tags
    out_df["model_text"] = combined
    out_df.attrs["selected_segment_names"] = selected_names
    return out_df


class TextOnlyClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


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


def encode_text(train_texts: list[str], test_texts: list[str], model_name: str, encode_batch_size: int):
    encoder = SentenceTransformer(model_name, device=str(DEVICE))
    train_embeddings = encoder.encode(
        train_texts,
        batch_size=encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    test_embeddings = encoder.encode(
        test_texts,
        batch_size=encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    return train_embeddings, test_embeddings


def create_dataloader(features: np.ndarray, labels: np.ndarray, shuffle: bool, batch_size: int):
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
    df = build_segment_text(df, args)
    selected_segment_names = df.attrs["selected_segment_names"]

    print("Selected text segments:", ", ".join(selected_segment_names))
    train_df, test_df = split_data(df, args.test_size)

    train_embeddings, test_embeddings = encode_text(
        train_texts=train_df["model_text"].tolist(),
        test_texts=test_df["model_text"].tolist(),
        model_name=args.bert_model_name,
        encode_batch_size=args.text_encode_batch_size,
    )

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    train_loader = create_dataloader(train_embeddings, y_train, shuffle=True, batch_size=args.batch_size)
    test_loader = create_dataloader(test_embeddings, y_test, shuffle=False, batch_size=args.batch_size)

    model = TextOnlyClassifier(
        input_dim=train_embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
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
    test_preds = (test_probs >= args.threshold).astype(int)

    print(f"\nBest ROC-AUC during training: {best_auc:.4f}")
    print(f"Selected text segments: {', '.join(selected_segment_names)}")
    print(f"Data path: {args.data_path}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(test_labels, test_probs):.4f}")
    print(f"Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
    print(f"Macro F1: {f1_score(test_labels, test_preds, average='macro'):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(test_labels, test_preds, digits=4))


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
