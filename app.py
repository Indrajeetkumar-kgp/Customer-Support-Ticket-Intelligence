"""Streamlit interface for ticket intent prediction and similar-case search."""

from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.retrieval import SemanticIndex

MODEL_DIR = Path("artifacts/transformer/model")
RETRIEVAL_DIR = Path("artifacts/retrieval")

st.set_page_config(page_title="Support Ticket Intelligence", page_icon="🎫", layout="wide")
st.title("Customer Support Ticket Intelligence")
st.caption("Intent predictions and similar examples for human review; no response is generated or sent.")
st.info("The model is trained on synthetic data. Treat its predictions as suggestions, not production performance claims.")

if not MODEL_DIR.exists() or not RETRIEVAL_DIR.exists():
    st.warning("Model artifacts are not ready yet. Follow the training and retrieval commands in the README.")
    st.stop()


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_index():
    return SemanticIndex(RETRIEVAL_DIR)


tokenizer, model = load_model()
index = load_index()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

message = st.text_area("Customer message", placeholder="For example: I need help changing the delivery address for my order.", height=130)
top_k = st.slider("Similar examples to retrieve", min_value=1, max_value=10, value=5)

if st.button("Analyze message", type="primary", disabled=not message.strip()):
    inputs = tokenizer(message, return_tensors="pt", truncation=True, max_length=128)
    with torch.inference_mode():
        logits = model(**{key: value.to(device) for key, value in inputs.items()}).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu()
    predicted_id = int(probabilities.argmax())
    intent = model.config.id2label[predicted_id]
    confidence = float(probabilities[predicted_id])

    left, right = st.columns(2)
    left.metric("Predicted intent", intent)
    right.metric("Model confidence", f"{confidence:.1%}")
    st.subheader("Similar training messages")
    results = index.search(message, top_k=top_k).rename(
        columns={"instruction": "Similar message", "intent": "Known intent", "similarity": "Cosine similarity"}
    )
    results["Cosine similarity"] = results["Cosine similarity"].map(lambda value: f"{value:.3f}")
    st.dataframe(results, use_container_width=True, hide_index=True)

