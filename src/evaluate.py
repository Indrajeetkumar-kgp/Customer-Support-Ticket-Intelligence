"""Evaluate trained models on the deterministic held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import LABEL_COLUMN, TEXT_COLUMN, load_support_data, make_splits
from src.metrics import classification_metrics
from src.train_transformer import TicketDataset


def evaluate_baseline(test: pd.DataFrame, output: Path) -> None:
    model = joblib.load("artifacts/baseline/model.joblib")
    truth = test[LABEL_COLUMN].tolist()
    predictions = model.predict(test[TEXT_COLUMN]).tolist()
    save_report("baseline", truth, predictions, output)


def evaluate_transformer(test: pd.DataFrame, output: Path) -> None:
    model_dir = Path("artifacts/transformer/model")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    labels = test[LABEL_COLUMN].tolist()
    dataset = TicketDataset(test[TEXT_COLUMN], pd.Series([0] * len(test)), tokenizer)
    loader = DataLoader(dataset, batch_size=32)
    predicted_ids = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None)
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            predicted_ids.extend(logits.argmax(dim=1).cpu().tolist())
    predictions = [model.config.id2label[index] for index in predicted_ids]
    save_report("transformer", labels, predictions, output)


def save_report(name: str, truth: list[str], predictions: list[str], output: Path) -> None:
    metrics = classification_metrics(truth, predictions)
    (output / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    labels = sorted(set(truth) | set(predictions))
    matrix = confusion_matrix(truth, predictions, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output / f"{name}_confusion_matrix.csv")
    pd.DataFrame({"true_intent": truth, "predicted_intent": predictions}).to_csv(
        output / f"{name}_predictions.csv", index=False
    )
    print(f"{name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Optional path to a downloaded dataset CSV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=("baseline", "transformer", "both"), default="both")
    args = parser.parse_args()
    _, _, test = make_splits(load_support_data(args.csv), seed=args.seed)
    output = Path("artifacts/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    if args.model in ("baseline", "both"):
        evaluate_baseline(test, output)
    if args.model in ("transformer", "both"):
        evaluate_transformer(test, output)


if __name__ == "__main__":
    main()

