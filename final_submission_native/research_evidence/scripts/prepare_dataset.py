
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Configuration
DATASET_NAME = "Podric/prowl-secrets-corpus"
OUTPUT_DIR = Path("data")
RANDOM_STATE = 42


def load_corpus():
    """Load the Prowl Secrets Corpus from Hugging Face."""
    print("Loading Prowl Secrets Corpus...")

    dataset = load_dataset(DATASET_NAME)
    print("Available splits:", list(dataset.keys()))

    # This corpus is expected to provide a train split.
    if "train" not in dataset:
        raise ValueError(
            "No train split found. Check the dataset configuration."
        )

    df = dataset["train"].to_pandas()
    print(f"Loaded {len(df):,} records.")
    return df


def prepare_code_data(df):
    """Keep labelled code examples and useful metadata."""

    required = {"text", "label_binary", "source"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Expected columns are missing: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    # Select code snippets only
    df = df[
        df["source"].astype(str).str.lower() == "code"
    ].copy()

    # Preserve metadata for auditing and leakage checks
    keep_columns = [
        "text",
        "label_binary",
        "source",
        "origin",
        "value",
        "label_type",
        "span",
        "context",
    ]

    df = df[
        [column for column in keep_columns if column in df.columns]
    ].copy()

    # Remove missing and empty snippets
    df = df.dropna(subset=["text", "label_binary"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    # Validate binary labels
    df["label_binary"] = pd.to_numeric(
        df["label_binary"], errors="coerce"
    )
    df = df[df["label_binary"].isin([0, 1])]
    df["label_binary"] = df["label_binary"].astype(int)

    # Remove exact duplicate snippets
    df = df.drop_duplicates(subset=["text"])

    # Use simpler names for the ML pipeline
    df = df.rename(
        columns={
            "text": "code",
            "label_binary": "label",
        }
    )

    return df.reset_index(drop=True)


def split_data(df):
    """Create stratified 70/15/15 splits."""

    if df["label"].nunique() != 2:
        raise ValueError("Both labels (0 and 1) are required.")

    if df["label"].value_counts().min() < 3:
        raise ValueError(
            "Not enough examples in one class to create all splits."
        )

    train_df, temporary_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary_df["label"],
    )

    return train_df, validation_df, test_df


def save_data(train_df, validation_df, test_df):
    """Save metadata-rich and model-ready CSV files."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    for name, split in splits.items():
        # Preserve metadata for research and auditing
        split.to_csv(
            OUTPUT_DIR / f"{name}.csv",
            index=False,
        )

        # Only code and target label for model input
        split[["code", "label"]].to_csv(
            OUTPUT_DIR / f"{name}_model.csv",
            index=False,
        )

        print(
            f"{name.title()}: {len(split):,} examples | "
            f"labels: {split['label'].value_counts().to_dict()}"
        )

    print(f"\nCSV files saved in: {OUTPUT_DIR.resolve()}")


def main():
    df = load_corpus()

    print("\nFiltering and cleaning code snippets...")
    df = prepare_code_data(df)

    if df.empty:
        raise ValueError("No usable code snippets remain.")

    print(f"Unique code snippets: {len(df):,}")
    print("\nOverall label distribution:")
    print(df["label"].value_counts().sort_index())

    train_df, validation_df, test_df = split_data(df)
    save_data(train_df, validation_df, test_df)


if __name__ == "__main__":
    main()
