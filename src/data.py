"""Dataset loading, normalization, and leakage-aware splitting."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
TEXT_COLUMN = "instruction"
LABEL_COLUMN = "intent"


def normalize_text(text: str) -> str:
    """Normalize text for duplicate grouping without changing model input."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def load_support_data(local_csv: str | Path | None = None) -> pd.DataFrame:
    """Load Bitext data from a local CSV or the Hugging Face Datasets Hub."""
    if local_csv:
        frame = pd.read_csv(local_csv)
    else:
        dataset = load_dataset(DATASET_ID, split="train")
        frame = dataset.to_pandas()

    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame[[TEXT_COLUMN, LABEL_COLUMN]].dropna().copy()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(str).str.strip()
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str).str.strip()
    frame = frame[(frame[TEXT_COLUMN] != "") & (frame[LABEL_COLUMN] != "")]
    frame["group_key"] = frame[TEXT_COLUMN].map(normalize_text)
    return frame.reset_index(drop=True)


def split_without_duplicate_leakage(
    frame: pd.DataFrame, test_size: float = 0.15, validation_size: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratify unique messages, then assign every duplicate to the same split."""
    group_labels = frame.groupby("group_key")[LABEL_COLUMN].nunique()
    inconsistent = group_labels[group_labels > 1]
    if not inconsistent.empty:
        raise ValueError("A normalized duplicate message maps to multiple intents; inspect these labels.")

    groups = frame[["group_key", LABEL_COLUMN]].drop_duplicates("group_key")
    counts = groups[LABEL_COLUMN].value_counts()
    rare = counts[counts < 3]
    if not rare.empty:
        raise ValueError(
            "Need at least three distinct messages per intent for stratified train/validation/test "
            f"splits. Rare intents: {rare.to_dict()}"
        )

    train_groups, holdout_groups = train_test_split(
        groups,
        test_size=test_size + validation_size,
        random_state=seed,
        stratify=groups[LABEL_COLUMN],
    )
    relative_validation = validation_size / (test_size + validation_size)
    validation_groups, test_groups = train_test_split(
        holdout_groups,
        test_size=1 - relative_validation,
        random_state=seed,
        stratify=holdout_groups[LABEL_COLUMN],
    )

    mapping: dict[str, str] = {}
    for name, subset in (("train", train_groups), ("validation", validation_groups), ("test", test_groups)):
        mapping.update({key: name for key in subset["group_key"]})
    tagged = frame.copy()
    tagged["split"] = tagged["group_key"].map(mapping)
    return tuple(tagged[tagged["split"] == name].reset_index(drop=True) for name in ("train", "validation", "test"))


def make_splits(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return split_without_duplicate_leakage(frame, seed=seed)

