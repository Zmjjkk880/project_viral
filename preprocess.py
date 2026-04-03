from pathlib import Path
import argparse

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_PATH = BASE_DIR / "data/raw/youtube_shorts_tiktok_trends_2025.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data/processed"


TEXT_COLUMNS = ["title", "hashtag", "title_keywords", "tags"]
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
NUMERIC_COLUMNS = [
    "upload_hour",
    "week_of_year",
    "is_weekend",
    "duration_sec",
    "creator_avg_views",
    "title_length",
    "has_emoji",
]
TARGET_COLUMNS = ["views", "engagement_rate"]


ALL_REQUIRED_COLUMNS = TEXT_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + TARGET_COLUMNS


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dataset for views/engagement regression.")
    parser.add_argument("--output-filename", type=str, default="processed_data_regression.csv")
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing_columns = [col for col in ALL_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def build_text_feature(df: pd.DataFrame) -> pd.Series:
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str)

    text_feature = (
        "[TITLE] "
        + df["title"].str.strip()
        + " [HASHTAG] "
        + df["hashtag"].str.strip()
        + " [KEYWORDS] "
        + df["title_keywords"].str.strip()
        + " [TAGS] "
        + df["tags"].str.strip()
    )
    return text_feature.str.replace(r"\s+", " ", regex=True).str.strip()


def preprocess_data(output_filename: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw data shape: {df.shape}")

    validate_columns(df)

    processed_df = df[TEXT_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + TARGET_COLUMNS].copy()

    processed_df["text"] = build_text_feature(processed_df)

    # Ensure categorical columns contain no missing values
    for col in CATEGORICAL_COLUMNS:
        if processed_df[col].isna().any():
            raise ValueError(f"Missing values found in categorical column: {col}")
        processed_df[col] = processed_df[col].astype(str)

    # Ensure numeric columns contain valid numeric values and no missing values
    for col in NUMERIC_COLUMNS:
        processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")
        if processed_df[col].isna().any():
            raise ValueError(f"Missing or invalid numeric values found in column: {col}")

    # Ensure regression targets are valid numeric values and no missing values
    for col in TARGET_COLUMNS:
        processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")
        if processed_df[col].isna().any():
            raise ValueError(f"Missing or invalid target values found in column: {col}")

    # Stabilize long-tail target for regression.
    processed_df["views"] = np.log1p(processed_df["views"])

    processed_df = processed_df.drop(columns=TEXT_COLUMNS)

    processed_data_path = PROCESSED_DATA_DIR / output_filename

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_data_path, index=False)

    print(f"Processed data shape: {processed_df.shape}")
    print(f"Saved processed data to: {processed_data_path}")
    print("Regression targets summary:")
    for target in TARGET_COLUMNS:
        print(
            f"{target}: min={processed_df[target].min():.4f}, "
            f"max={processed_df[target].max():.4f}, "
            f"mean={processed_df[target].mean():.4f}"
        )
    print("Note: 'views' has been transformed with log1p (log(views + 1)).")

    return processed_df


if __name__ == "__main__":
    args = parse_args()
    preprocess_data(output_filename=args.output_filename)
