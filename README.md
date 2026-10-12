# Customer Support Ticket Intelligence

A portfolio project that uses deep learning and transformer models to help support teams route customer messages and find relevant past cases.

## Project goals

Given a support message, the system will:

1. Predict its support **intent** with a fine-tuned transformer.
2. Retrieve semantically similar messages to help an agent find related cases.
3. Show prediction confidence and evaluation details in a small interactive demo.

The first version focuses on intent classification and retrieval. Urgency prediction can be added later if suitable labeled data is available.

## Why this project

This project demonstrates an end-to-end data science workflow: data inspection, text preprocessing, a classical baseline, transformer fine-tuning, error analysis, semantic retrieval, and a lightweight interface. It is designed as decision support for human agents; it does not generate or send customer responses.

## Dataset

The project uses Bitext's [Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), available on Hugging Face. The dataset card currently reports about 26.9K rows and 27 intent labels, with `instruction` as the input text and `intent` as the target.

**Data note:** Bitext describes this as synthetic data. Results on it are useful for a reproducible portfolio experiment, but should not be presented as evidence of performance on real customer conversations. The dataset page lists the CDLA-Sharing-1.0 license; review its terms and the dataset card before use.

## Modeling plan

### 1. Establish a baseline

Train a TF-IDF + logistic regression classifier using the same train/validation/test split as the transformer. This provides a clear reference point and helps identify whether the added model complexity is worthwhile.

### 2. Fine-tune a transformer

Fine-tune a pretrained DistilBERT sequence-classification model to predict the dataset's intent label. Keep preprocessing, splits, and evaluation consistent with the baseline. Track the selected model, tokenizer, random seed, and training configuration.

### 3. Add semantic retrieval

Encode the training messages with a sentence-embedding model and retrieve the top-k most similar examples for a new message. Display the matched text and its known intent as context for the user.

### 4. Analyze errors

Review examples by class, especially intents with low recall or frequent confusion. Check for class imbalance, duplicate or near-duplicate messages across splits, and data leakage before reporting metrics.

## Evaluation

Report results on a held-out test set. Include:

- Macro-F1 as the primary classification metric.
- Accuracy and weighted-F1 for context.
- Per-class precision, recall, and F1, plus a confusion matrix.
- Baseline-versus-transformer comparison.
- Retrieval quality, using manually reviewed examples or a labeled relevance set if one is created.
- Inference latency and model size if deployment efficiency is discussed.

Do not fill in results until the experiments have been run. Suggested reporting table:

| Model / component | Metric | Test result |
|---|---|---:|
| TF-IDF + logistic regression | Macro-F1 | TBD |
| Fine-tuned DistilBERT | Macro-F1 | TBD |
| Semantic retrieval | Retrieval evaluation | TBD |

## Suggested repository layout

```text
customer-support-ticket-intelligence/
├── app.py
├── data/
│   └── README.md              # dataset instructions; keep raw data out of Git
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   ├── evaluate.py
│   └── retrieval.py
├── artifacts/                 # saved models and reports; usually gitignored
├── requirements.txt
└── README.md
```

## Getting started

The commands below are a suggested setup. Adjust them to match the files and scripts in your implementation.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

Load the dataset from Hugging Face in your data preparation code, or follow the dataset card's current instructions. Keep downloaded data and trained model artifacts out of version control unless there is a clear reason to include them.

Run the project commands from the repository root:

```bash
python -m src.train_baseline
python -m src.train_transformer
python -m src.retrieval
python -m src.evaluate
streamlit run app.py
```

The baseline and transformer training commands create a duplicate-aware, stratified split targeting 70/15/15 of the unique normalized messages. Exact normalized duplicates are assigned to the same split to reduce leakage, so row percentages may differ. Both training scripts use seed `42` by default; pass `--seed` to all commands if you change it. Evaluation expects both trained models by default; use `python -m src.evaluate --model baseline` or `--model transformer` to evaluate one.

## Demo

The demo accepts a message and displays:

- Predicted intent and confidence.
- The top-k similar training examples and their similarity scores.
- A short note that the dataset is synthetic and predictions need human review.

### Illustrative prediction

The following shows the format only. The intent, confidence, similarity scores, and retrieved examples must come from an actual model run; this is not a measured result.

```text
Customer message: I need help cancelling an order.
Predicted intent: [model prediction]
Confidence: [model confidence]

Similar training messages:
1. [retrieved message] — [known intent] — similarity [score]
2. [retrieved message] — [known intent] — similarity [score]
3. [retrieved message] — [known intent] — similarity [score]
```

### Files produced

After running the project commands, the `artifacts/` directory contains:

- `baseline/model.joblib` and `baseline/split_info.json` from baseline training.
- `transformer/model/` and `transformer/metadata.json` from fine-tuning.
- `retrieval/embeddings.npy`, `retrieval/examples.csv`, and the embedding model name from index creation.
- `evaluation/*_metrics.json`, `evaluation/*_confusion_matrix.csv`, and `evaluation/*_predictions.csv` from evaluation.

These generated files are gitignored; the README and source code are the reproducible project deliverables.

## Reproducibility and responsible use

- Use a fixed random seed and save the data split identifiers.
- Prevent duplicate or near-duplicate examples from leaking across splits.
- Document label definitions, preprocessing, model checkpoint, and hyperparameters.
- Do not expose personal or sensitive information in logs or screenshots.
- Treat predictions and retrieved examples as suggestions for a human reviewer.

## Results

Add a brief summary here after completing the experiments. Evaluation reports are written under `artifacts/evaluation/`. A good report compares the transformer with the baseline on the same held-out split, highlights class-level errors, and describes retrieval quality. Do not assume the transformer wins or report a target score before running the experiment.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| TF-IDF + logistic regression | TBD | TBD | TBD |
| Fine-tuned DistilBERT | TBD | TBD | TBD |

- Best model and test metrics: **TBD**
- Improvement over baseline: **TBD**
- Main error patterns: **TBD**
- Retrieval observations: **TBD**

## Build stages

1. Install dependencies and load the dataset (`src/data.py`).
2. Train the TF-IDF baseline (`src/train_baseline.py`).
3. Fine-tune DistilBERT (`src/train_transformer.py`).
4. Build the semantic retrieval index (`src/retrieval.py`).
5. Evaluate held-out results (`src/evaluate.py`).
6. Start the interactive demo (`app.py`).

## References

- [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- [Hugging Face Transformers sequence classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)
