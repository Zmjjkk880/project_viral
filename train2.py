from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data/processed/processed_data_regression.csv"

TARGET_COLUMNS = ["views", "engagement_rate"]
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
BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

NUMERIC_COLUMNS = [
    "upload_hour",
    "week_of_year",
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
    parser = argparse.ArgumentParser(description="Train MLP for views/engagement regression.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument(
        "--text-encoder",
        type=str,
        default="auto",
        choices=["auto", "sbert", "tfidf"],
    )
    parser.add_argument("--tfidf-max-features", type=int, default=2000)
    parser.add_argument(
        "--token-mixer",
        type=str,
        default="raw_concat",
        choices=["raw_concat", "projected_concat", "weighted_sum", "attention_pool"],
    )
    parser.add_argument("--proj-dim", type=int, default=64)
    return parser.parse_args()


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

        if token_mixer == "raw_concat":
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


class MLPRegressor(nn.Module):
    def __init__(
        self,
        text_dim: int,
        tab_dim: int,
        hidden_dim: int,
        dropout: float,
        token_mixer: str,
        proj_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.token_mixer = TokenMixer(
            text_dim=text_dim,
            tab_dim=tab_dim,
            token_mixer=token_mixer,
            proj_dim=proj_dim,
            dropout=dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.token_mixer.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.token_mixer(x)
        return self.regressor(mixed)


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame, test_size: float):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
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


def _encode_text_sbert(train_df: pd.DataFrame, test_df: pd.DataFrame):
    encoder = SentenceTransformer(BERT_MODEL_NAME, device=str(DEVICE))
    train_text = encoder.encode(
        train_df[TEXT_COLUMN].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    test_text = encoder.encode(
        test_df[TEXT_COLUMN].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    return train_text, test_text


def _encode_text_tfidf(train_df: pd.DataFrame, test_df: pd.DataFrame, max_features: int):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    train_text = vectorizer.fit_transform(train_df[TEXT_COLUMN].tolist()).toarray().astype(np.float32)
    test_text = vectorizer.transform(test_df[TEXT_COLUMN].tolist()).toarray().astype(np.float32)
    return train_text, test_text


def encode_text(train_df: pd.DataFrame, test_df: pd.DataFrame, text_encoder: str, tfidf_max_features: int):
    if text_encoder == "sbert":
        return _encode_text_sbert(train_df, test_df)
    if text_encoder == "tfidf":
        return _encode_text_tfidf(train_df, test_df, tfidf_max_features)

    try:
        train_text, test_text = _encode_text_sbert(train_df, test_df)
        print("Text encoder: SBERT")
        return train_text, test_text
    except Exception as exc:
        print(f"SBERT unavailable, fallback to TF-IDF. Reason: {exc}")
        return _encode_text_tfidf(train_df, test_df, tfidf_max_features)


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame, text_encoder: str, tfidf_max_features: int):
    train_text, test_text = encode_text(train_df, test_df, text_encoder=text_encoder, tfidf_max_features=tfidf_max_features)
    train_tab, test_tab = prepare_tabular_features(train_df, test_df)

    x_train = np.hstack([train_text, train_tab]).astype(np.float32)
    x_test = np.hstack([test_text, test_tab]).astype(np.float32)

    y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)

    print(f"Train feature shape: {x_train.shape}")
    print(f"Test feature shape: {x_test.shape}")
    print(f"Train target shape: {y_train.shape}")
    print(f"Test target shape: {y_test.shape}")

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
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            preds = model(batch_x)
            loss = criterion(preds, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.vstack(all_preds), np.vstack(all_labels)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    metrics = {}

    for idx, target_name in enumerate(TARGET_COLUMNS):
        true_vals = y_true[:, idx]
        pred_vals = y_pred[:, idx]
        metrics[target_name] = {
            "mae": mean_absolute_error(true_vals, pred_vals),
            "rmse": np.sqrt(mean_squared_error(true_vals, pred_vals)),
            "r2": r2_score(true_vals, pred_vals),
        }

    metrics["overall"] = {
        "mae": float(np.mean([metrics[t]["mae"] for t in TARGET_COLUMNS])),
        "rmse": float(np.mean([metrics[t]["rmse"] for t in TARGET_COLUMNS])),
        "r2": float(np.mean([metrics[t]["r2"] for t in TARGET_COLUMNS])),
    }

    return metrics


def train_model(args):
    set_seed(RANDOM_STATE)

    df = load_data(args.data_path)
    train_df, test_df = split_data(df, args.test_size)

    x_train, x_test, y_train, y_test = build_features(
        train_df,
        test_df,
        text_encoder=args.text_encoder,
        tfidf_max_features=args.tfidf_max_features,
    )

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    train_loader = create_dataloader(x_train, y_train_scaled, shuffle=True, batch_size=args.batch_size)
    test_loader = create_dataloader(x_test, y_test_scaled, shuffle=False, batch_size=args.batch_size)

    text_dim = x_train.shape[1] - len(NUMERIC_COLUMNS) - len(
        pd.get_dummies(train_df[CATEGORICAL_COLUMNS], drop_first=False).columns
    )
    tab_dim = x_train.shape[1] - text_dim

    model = MLPRegressor(
        text_dim=text_dim,
        tab_dim=tab_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        token_mixer=args.token_mixer,
        proj_dim=args.proj_dim,
        output_dim=len(TARGET_COLUMNS),
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_score = float("-inf")
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_x.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        test_loss, test_preds_scaled, test_labels_scaled = evaluate(model, test_loader, criterion)
        test_preds = y_scaler.inverse_transform(test_preds_scaled)
        test_labels = y_scaler.inverse_transform(test_labels_scaled)
        test_metrics = compute_regression_metrics(test_labels, test_preds)

        test_score = test_metrics["overall"]["r2"]
        if test_score > best_score:
            best_score = test_score
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss(MSE scaled): {avg_train_loss:.4f} | "
            f"Test Loss(MSE scaled): {test_loss:.4f} | "
            f"Test Mean R2: {test_metrics['overall']['r2']:.4f}"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_preds_scaled, test_labels_scaled = evaluate(model, test_loader, criterion)
    test_preds = y_scaler.inverse_transform(test_preds_scaled)
    test_labels = y_scaler.inverse_transform(test_labels_scaled)
    final_metrics = compute_regression_metrics(test_labels, test_preds)

    print(f"\nBest validation mean R2 during training: {best_score:.4f}")
    print(f"Final Test Loss (MSE on scaled targets): {test_loss:.4f}")

    print("\nConfig:")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Dropout: {args.dropout}")
    print(f"Test size: {args.test_size}")
    print(f"Data path: {args.data_path}")
    print(f"Text encoder: {args.text_encoder}")
    print(f"TF-IDF max features: {args.tfidf_max_features}")
    print(f"Token mixer: {args.token_mixer}")
    print(f"Projection dim: {args.proj_dim}")

    print("\nRegression Metrics:")
    for target_name in TARGET_COLUMNS:
        target_metrics = final_metrics[target_name]
        print(
            f"{target_name}: "
            f"MAE={target_metrics['mae']:.4f}, "
            f"RMSE={target_metrics['rmse']:.4f}, "
            f"R2={target_metrics['r2']:.4f}"
        )

    print(
        "Overall (mean across targets): "
        f"MAE={final_metrics['overall']['mae']:.4f}, "
        f"RMSE={final_metrics['overall']['rmse']:.4f}, "
        f"R2={final_metrics['overall']['r2']:.4f}"
    )


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
