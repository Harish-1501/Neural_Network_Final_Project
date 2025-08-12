# app.py — Streamlit front-end for your saved Keras model (best.keras)
# 
# Features
# - Loads model from ./models/best.keras (compile=False for fast inference)
# - Loads tokenizer from ./models/tokenizer.json or ./models/tokenizer.zip
# - Single-text and batch CSV prediction
# - Adjustable decision threshold + customizable label names
# - Clean UI with helpful tips and downloadable results
#
# Expected model I/O (from your uploaded model):
#   Input:  int32 token IDs, shape [batch, 512], pad/truncate with 0s
#   Output: float32 sigmoid probability for the positive class, shape [batch, 1]
#
# Project structure suggestion:
#   models/
#     best.keras
#     tokenizer.json   (or tokenizer.zip containing tokenizer.json)
#   app.py
#   requirements.txt   (tensorflow, streamlit, numpy, pandas)


from __future__ import annotations

import io
import json
import os
import zipfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# TensorFlow / Keras
# Prefer Keras 3 loader if available; fall back to tf.keras
try:
    import keras  # Keras 3.x
    KERAS3_AVAILABLE = True
except Exception:
    KERAS3_AVAILABLE = False

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# -----------------------------
# Config
# -----------------------------
MODEL_PATH_CANDIDATES = [
    "models/best.keras",
    "best.keras",
]
TOKENIZER_PATH_CANDIDATES = [
    "models/tokenizer.json",
    "models/tokenizer.zip",
    "tokenizer.json",
    "tokenizer.zip",
]
MAX_LEN = 512    # required by your model
PAD_VALUE = 0    # model masks zeros

# Default label names (customizable in sidebar)
DEFAULT_NEG_LABEL = "negative"
DEFAULT_POS_LABEL = "positive"

# -----------------------------
# Page setup & styles
# -----------------------------
st.set_page_config(
    page_title="Text Classifier • best.keras",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
      /* App-wide polish */
      .prob-bar > div > div { transition: width 400ms ease-in-out; }
      .small-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.85rem; color: #666; }
      .ok { color: #16a34a; }
      .warn { color: #b45309; }
      .bad { color: #dc2626; }
      .muted { color: #6b7280; }
      .tight { line-height: 1.2; }
      .chip { display:inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; background:#f3f4f6; margin-left: .35rem; font-size:.8rem; }
      .hsep { height: 1px; background: #eee; margin: .5rem 0 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Toxicity Prediction")
st.caption(
    "Feed raw text → we apply your saved tokenizer → model returns a positive-class probability (sigmoid)."
)

# -----------------------------
# Utility: find first existing path
# -----------------------------
def _first_existing(paths: List[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# -----------------------------
# Load tokenizer (JSON or inside ZIP)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_tokenizer():
    tok_path = _first_existing(TOKENIZER_PATH_CANDIDATES)
    if tok_path is None:
        raise FileNotFoundError(
            "Tokenizer not found. Expected tokenizer.json or tokenizer.zip in ./models or project root."
        )

    if tok_path.endswith(".json"):
        with open(tok_path, "r", encoding="utf-8") as f:
            tok_json = f.read()
        tokenizer = tokenizer_from_json(tok_json)
        src = os.path.basename(tok_path)
    elif tok_path.endswith(".zip"):
        with zipfile.ZipFile(tok_path, "r") as zf:
            names = zf.namelist()
            # Prefer a file literally named tokenizer.json
            json_name = None
            for n in names:
                if n.endswith("tokenizer.json"):
                    json_name = n
                    break
            if json_name is None:
                # fallback to any .json inside zip
                cand = [n for n in names if n.lower().endswith(".json")]
                if not cand:
                    raise FileNotFoundError("No .json found inside tokenizer zip.")
                json_name = cand[0]
            tok_json = zf.read(json_name).decode("utf-8")
        tokenizer = tokenizer_from_json(tok_json)
        src = f"{os.path.basename(tok_path)} → {json_name}"
    else:
        raise ValueError("Unsupported tokenizer format. Use .json or .zip.")

    return tokenizer, src

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = _first_existing(MODEL_PATH_CANDIDATES)
    if model_path is None:
        raise FileNotFoundError(
            "Model not found. Put best.keras in ./models (or project root)."
        )
    # Try Keras 3 first (your file was saved with Keras 3 serialization)
    if KERAS3_AVAILABLE:
        try:
            import keras
            model = keras.saving.load_model(model_path, compile=False)
            return model, os.path.basename(model_path)
        except Exception as e:
            # Fall through to tf.keras fallback
            st.warning(f"Keras 3 loader failed: {e}. Falling back to tf.keras…")
    # Fallback: tf.keras loader (works for models saved with tf.keras / Keras 2)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model, os.path.basename(model_path)

# -----------------------------
# Preprocess & predict helpers
# -----------------------------
def preprocess_texts(texts: List[str], tokenizer, max_len: int = MAX_LEN) -> np.ndarray:
    # Convert raw text → token IDs
    seqs = tokenizer.texts_to_sequences(texts)
    # Pad/truncate to model's expected length
    x = pad_sequences(
        seqs,
        maxlen=max_len,
        padding="post",
        truncating="post",
        value=PAD_VALUE,
    )
    return x.astype("int32")

@st.cache_resource(show_spinner=False)
def _bootstrap():
    tokenizer, tokenizer_src = load_tokenizer()
    model, model_src = load_model()
    return tokenizer, tokenizer_src, model, model_src

# Initialize resources
err_container = st.empty()
try:
    tokenizer, tokenizer_src, model, model_src = _bootstrap()
except Exception as e:
    err_container.error(f"❌ Startup error: {e}")
    st.stop()

with st.expander("Runtime status", expanded=False):
    st.markdown(
        f"**Model:** `{model_src}`  \n"
        f"**Tokenizer:** `{tokenizer_src}`  \n"
        f"**Max length:** `{MAX_LEN}` · **Pad value:** `{PAD_VALUE}`"
    )

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Settings")
thresh = st.sidebar.slider("Decision threshold (Recommended : 0.6)", 0.0, 1.0, 0.50, 0.01)

# thresh = st.sidebar.number_input(
#     "Decision threshold (≥ → positive)",
#     min_value=0.0,
#     max_value=1.0,
#     value=float(0.0487249419093132),
#     step=0.0001,
#     format="%.6f",
# )


neg_label = st.sidebar.text_input("Negative label", value=DEFAULT_NEG_LABEL)
pos_label = st.sidebar.text_input("Positive label", value=DEFAULT_POS_LABEL)

st.sidebar.caption(
    "Tip: if you tuned a best threshold on validation (e.g., max-F1), set it here."
)

# -----------------------------
# Single text prediction
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("🔮 Single-text prediction")
    with st.form("single_form", clear_on_submit=False):
        text = st.text_area(
            "Enter text",
            height=140,
            placeholder="Type or paste your text here…",
        )
        submitted = st.form_submit_button("Predict")

    if submitted:
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Tokenizing and running the model…"):
                X = preprocess_texts([text], tokenizer, MAX_LEN)
                prob =(1- float(model.predict(X, verbose=0)[0][0]))
            label = pos_label if prob >= thresh else neg_label

            st.markdown(f"**Prediction:** `{label}`  <span class='chip'>{prob:.4f}</span>", unsafe_allow_html=True)
            st.progress(prob, text=f"Positive probability: {prob:.2%}")
            st.caption(
                "Probability is the model's sigmoid output for the positive class."
            )

# -----------------------------
# Batch CSV prediction
# -----------------------------
with right:
    st.subheader("📦 Batch prediction (CSV)")
    csv_file = st.file_uploader("Upload a CSV", type=["csv"], accept_multiple_files=False)

    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            csv_file.seek(0)
            df = pd.read_csv(csv_file, encoding_errors="ignore")

        st.write("Preview:")
        st.dataframe(df.head(10), use_container_width=True)

        # Column selection
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            st.error("No text columns found in the CSV.")
        else:
            col = st.selectbox("Select the text column", text_cols)
            run_batch = st.button("Run batch predictions")

            if run_batch:
                with st.spinner("Running predictions…"):
                    texts = df[col].fillna("").astype(str).tolist()
                    X = preprocess_texts(texts, tokenizer, MAX_LEN)
                    probs = (1- model.predict(X, verbose=0).reshape(-1))
                    labels = np.where(probs >= thresh, pos_label, neg_label)

                    out = df.copy()
                    out["probability"] = probs
                    out["label"] = labels

                st.success("Done!")
                st.dataframe(out.head(20), use_container_width=True)

                # Download
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button(
                    label="⬇️ Download predictions (CSV)",
                    data=buf.getvalue(),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

# -----------------------------
# Helpful notes
# -----------------------------
st.markdown(
    """
    <div class='hsep'></div>
    <div class='small-mono muted'>
      <strong>Notes</strong> · The model expects integer token IDs of length 512 padded with 0s. We use your saved tokenizer's vocabulary (OOV token supported).  
      If your inputs exceed 512 tokens, they will be truncated on the right (post).  
      For reproducible behavior across environments, use the same TensorFlow/Keras version used for training when possible.
    </div>
    """,
    unsafe_allow_html=True,
)
