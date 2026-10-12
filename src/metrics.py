"""Shared classification metrics."""

from sklearn.metrics import accuracy_score, classification_report, f1_score


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }

