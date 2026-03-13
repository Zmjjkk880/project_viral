from pathlib import Path
import argparse

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
TARGET_COLUMN = "viral"

LEAKAGE_COLUMN = "engagement_rate"
VIEWS_COLUMN = "views"


ALL_REQUIRED_COLUMNS = TEXT_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [LEAKAGE_COLUMN, VIEWS_COLUMN]


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess viral video dataset.")
    parser.add_argument("--views-threshold", type=int, default=100000)
    parser.add_argument("--engagement-threshold", type=float, default=0.06)
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


def create_viral_label(df: pd.DataFrame, views_threshold: int, engagement_threshold: float) -> pd.Series:
    print(
        f"Using viral definition: views >= {views_threshold} AND engagement_rate >= {engagement_threshold:.2f}"
    )

    viral = (
        (df[VIEWS_COLUMN] >= views_threshold) &
        (df[LEAKAGE_COLUMN] >= engagement_threshold)
    )

    return viral.astype(int)


def preprocess_data(views_threshold: int, engagement_threshold: float) -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw data shape: {df.shape}")

    validate_columns(df)

    processed_df = df[TEXT_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [LEAKAGE_COLUMN, VIEWS_COLUMN]].copy()

    processed_df["text"] = build_text_feature(processed_df)
    processed_df[TARGET_COLUMN] = create_viral_label(
        processed_df,
        views_threshold=views_threshold,
        engagement_threshold=engagement_threshold,
    )

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

    processed_df = processed_df.drop(columns=TEXT_COLUMNS + [LEAKAGE_COLUMN, VIEWS_COLUMN])

    output_filename = f"processed_data_{views_threshold}_{engagement_threshold:.2f}.csv"
    processed_data_path = PROCESSED_DATA_DIR / output_filename

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_data_path, index=False)

    print(f"Processed data shape: {processed_df.shape}")
    print(f"Saved processed data to: {processed_data_path}")

    label_distribution = processed_df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100
    non_viral_pct = label_distribution.get(0, 0.0)
    viral_pct = label_distribution.get(1, 0.0)

    print("Viral label distribution:")
    print(f"Class 0 (not viral): {non_viral_pct:.2f}%")
    print(f"Class 1 (viral): {viral_pct:.2f}%")

    return processed_df


if __name__ == "__main__":
    args = parse_args()
    preprocess_data(
        views_threshold=args.views_threshold,
        engagement_threshold=args.engagement_threshold,
    )