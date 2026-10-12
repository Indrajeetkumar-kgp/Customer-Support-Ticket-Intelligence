"""Train a TF-IDF + logistic regression reference model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import LABEL_COLUMN, TEXT_COLUMN, load_support_data, make_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Optional path to a downloaded dataset CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = load_support_data(args.csv)
    train, validation, test = make_splits(frame, seed=args.seed)
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(train[TEXT_COLUMN], train[LABEL_COLUMN])
    output = Path("artifacts/baseline")
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    split_info = {
        "seed": args.seed,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "unique_messages": {
            "train": int(train["group_key"].nunique()),
            "validation": int(validation["group_key"].nunique()),
            "test": int(test["group_key"].nunique()),
        },
        "classes": sorted(train[LABEL_COLUMN].unique().tolist()),
    }
    (output / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    print(f"Saved baseline to {output.resolve()}")
    print(json.dumps(split_info, indent=2))


if __name__ == "__main__":
    main()

