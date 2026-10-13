"""Fine-tune DistilBERT for customer support intent classification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data import LABEL_COLUMN, TEXT_COLUMN, load_support_data, make_splits

DEFAULT_MODEL = "distilbert-base-uncased"


class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 64):
        self.encodings = tokenizer(
            texts.tolist(), truncation=True, padding=False, max_length=max_length
        )
        self.labels = labels.to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = int(self.labels[index])
        return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Optional path to a downloaded dataset CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be at least 1")
    if not torch.cuda.is_available():
        torch.set_num_threads(args.cpu_threads)

    frame = load_support_data(args.csv)
    train, validation, test = make_splits(frame, seed=args.seed)
    labels = sorted(frame[LABEL_COLUMN].unique().tolist())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), label2id=label2id, id2label=id2label
    )

    train_set = TicketDataset(train[TEXT_COLUMN], train[LABEL_COLUMN].map(label2id), tokenizer, args.max_length)
    validation_set = TicketDataset(
        validation[TEXT_COLUMN], validation[LABEL_COLUMN].map(label2id), tokenizer, args.max_length
    )
    output = Path("artifacts/transformer")
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=0,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    def compute_metrics(prediction):
        logits, gold = prediction
        predicted = np.argmax(logits, axis=1)
        return {"macro_f1": f1_score(gold, predicted, average="macro", zero_division=0)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=validation_set,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output / "model"))
    tokenizer.save_pretrained(str(output / "model"))
    metadata = {
        "base_model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "cpu_threads": args.cpu_threads,
        "labels": labels,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved fine-tuned model to {(output / 'model').resolve()}")


if __name__ == "__main__":
    main()

