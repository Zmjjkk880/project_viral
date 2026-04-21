from pathlib import Path
import argparse

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import log as base_log


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data/processed/processed_data.csv"
TOKEN_MIXER_NAME = "tabular_projected_only"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an MLP for viral video prediction using tabular features projected to proj_dim."
    )
    parser.add_argument("--batch-size", type=int, default=base_log.DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=base_log.DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=base_log.DEFAULT_LEARNING_RATE)
    parser.add_argument("--hidden-dim", type=int, default=base_log.DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=base_log.DEFAULT_DROPOUT)
    parser.add_argument("--test-size", type=float, default=base_log.DEFAULT_TEST_SIZE)
    parser.add_argument("--threshold", type=float, default=base_log.DEFAULT_THRESHOLD)
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument(
        "--token-mixer",
        type=str,
        default=TOKEN_MIXER_NAME,
        choices=[TOKEN_MIXER_NAME],
        help="Kept for notebook compatibility. This script only supports tabular_projected_only.",
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
        help="Unsupported in this script. It trains a single tabular projected only model.",
    )
    return parser.parse_args()


class TabularProjectedOnlyMixer(nn.Module):
    def __init__(self, text_dim: int, tab_dim: int, proj_dim: int, dropout: float) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.tab_dim = tab_dim
        self.output_dim = proj_dim
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tab = x[:, self.text_dim :]
        return self.tab_proj(x_tab)

    def get_weighted_sum_weights(self, x: torch.Tensor):
        return None


class TabularProjectedOnlyClassifier(nn.Module):
    def __init__(
        self,
        text_dim: int,
        tab_dim: int,
        hidden_dim: int,
        dropout: float,
        proj_dim: int,
    ) -> None:
        super().__init__()
        self.token_mixer = TabularProjectedOnlyMixer(
            text_dim=text_dim,
            tab_dim=tab_dim,
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


def train_single_model(
    token_mixer: str,
    x_train,
    x_test,
    y_train,
    y_test,
    text_dim: int,
    tab_dim: int,
    args,
):
    if token_mixer != TOKEN_MIXER_NAME:
        raise ValueError(f"{TOKEN_MIXER_NAME} script only supports token_mixer={TOKEN_MIXER_NAME}")

    train_loader = base_log.create_dataloader(x_train, y_train, shuffle=True, batch_size=args.batch_size)
    test_loader = base_log.create_dataloader(x_test, y_test, shuffle=False, batch_size=args.batch_size)

    model = TabularProjectedOnlyClassifier(
        text_dim=text_dim,
        tab_dim=tab_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
    ).to(base_log.DEVICE)

    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(base_log.DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_auc = float("-inf")
    best_model_state = None

    print(f"\n===== Training token mixer: {token_mixer} =====")

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(base_log.DEVICE)
            batch_y = batch_y.to(base_log.DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_x.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        test_loss, test_probs_epoch, test_labels_epoch, _ = base_log.evaluate(model, test_loader, criterion)
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

    test_loss, test_probs, test_labels, _ = base_log.evaluate(model, test_loader, criterion)
    test_preds = (test_probs >= args.threshold).astype(int)
    test_auc = roc_auc_score(test_labels, test_probs)
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro")

    print(f"Best ROC-AUC during training ({token_mixer}): {best_auc:.4f}")
    print(f"Final Test Loss ({token_mixer}): {test_loss:.4f}")
    print(f"Final Test ROC-AUC ({token_mixer}): {test_auc:.4f}")
    print(f"Final Accuracy ({token_mixer}): {test_accuracy:.4f}")
    print(f"Final Macro F1 ({token_mixer}): {test_macro_f1:.4f}")

    return {
        "token_mixer": token_mixer,
        "test_loss": test_loss,
        "test_auc": test_auc,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "probs": test_probs,
        "preds": test_preds,
        "labels": test_labels,
        "weights": None,
    }


def train_model(args):
    if args.compare_all:
        raise ValueError("log_tabular_proj.py does not support --compare-all.")

    base_log.set_seed(base_log.RANDOM_STATE)

    df = base_log.load_data(args.data_path)
    train_df, test_df = base_log.split_data(df, args.test_size)
    x_train, x_test, y_train, y_test = base_log.build_features(train_df, test_df)

    text_dim = x_train.shape[1] - len(base_log.NUMERIC_COLUMNS) - len(
        pd.get_dummies(train_df[base_log.CATEGORICAL_COLUMNS], drop_first=False).columns
    )
    tab_dim = x_train.shape[1] - text_dim

    result = train_single_model(
        token_mixer=args.token_mixer,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        text_dim=text_dim,
        tab_dim=tab_dim,
        args=args,
    )

    print("\n===== Single Model Summary =====")
    print(f"Token mixer: {args.token_mixer}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Dropout: {args.dropout}")
    print(f"Test size: {args.test_size}")
    print(f"Prediction threshold: {args.threshold}")
    print(f"Data path: {args.data_path}")
    print(f"Projection dim: {args.proj_dim}")
    print(f"Test Loss: {result['test_loss']:.4f}")
    print(f"Test ROC-AUC: {result['test_auc']:.4f}")
    print(f"Accuracy: {result['test_accuracy']:.4f}")
    print(f"Macro F1: {result['test_macro_f1']:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
