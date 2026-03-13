from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_PATH = BASE_DIR / "data/raw/youtube_shorts_tiktok_trends_2025.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data/processed/processed_data.csv"


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


def create_viral_label(df: pd.DataFrame) -> pd.Series:
    print("Using viral definition: views >= 100000 AND engagement_rate >= 0.10")

    viral = (
        (df[VIEWS_COLUMN] >= 100000) &
        (df[LEAKAGE_COLUMN] >= 0.10)
    )

    return viral.astype(int)


def preprocess_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw data shape: {df.shape}")

    validate_columns(df)

    processed_df = df[TEXT_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [LEAKAGE_COLUMN, VIEWS_COLUMN]].copy()

    processed_df["text"] = build_text_feature(processed_df)
    processed_df[TARGET_COLUMN] = create_viral_label(processed_df)

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

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed data shape: {processed_df.shape}")
    print(f"Saved processed data to: {PROCESSED_DATA_PATH}")
    print("Viral label distribution:")
    print(processed_df[TARGET_COLUMN].value_counts(normalize=True))

    return processed_df


if __name__ == "__main__":
    preprocess_data()