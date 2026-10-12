"""Build and query a lightweight semantic index of training messages."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data import LABEL_COLUMN, TEXT_COLUMN, load_support_data, make_splits

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_index(
    csv_path: str | None = None, model_name: str = DEFAULT_EMBEDDING_MODEL, seed: int = 42
) -> None:
    train, _, _ = make_splits(load_support_data(csv_path), seed=seed)
    examples = train.drop_duplicates("group_key")[[TEXT_COLUMN, LABEL_COLUMN]].reset_index(drop=True)
    encoder = SentenceTransformer(model_name)
    vectors = encoder.encode(
        examples[TEXT_COLUMN].tolist(),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    output = Path("artifacts/retrieval")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", vectors.astype("float32"))
    examples.to_csv(output / "examples.csv", index=False)
    (output / "embedding_model.txt").write_text(model_name, encoding="utf-8")
    print(f"Saved {len(examples)} indexed messages to {output.resolve()}")


class SemanticIndex:
    def __init__(self, artifact_dir: str | Path = "artifacts/retrieval"):
        artifact_dir = Path(artifact_dir)
        self.examples = pd.read_csv(artifact_dir / "examples.csv")
        self.embeddings = np.load(artifact_dir / "embeddings.npy")
        model_name = (artifact_dir / "embedding_model.txt").read_text(encoding="utf-8").strip()
        self.encoder = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_vector = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scores = self.embeddings @ query_vector
        count = min(top_k, len(scores))
        best = np.argsort(scores)[::-1][:count]
        results = self.examples.iloc[best].copy().reset_index(drop=True)
        results.insert(0, "similarity", scores[best])
        return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Optional path to a downloaded dataset CSV")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_index(args.csv, args.model, args.seed)


if __name__ == "__main__":
    main()

