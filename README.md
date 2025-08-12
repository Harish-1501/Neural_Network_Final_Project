# YouTube Comment Toxicity Classifier (Streamlit + Keras BiLSTM)

A simple web app for classifying YouTube comments as **toxic / non‑toxic** using a trained **BiLSTM** model.  
Built with **Streamlit**, loads a saved model (`best.keras`) and tokenizer (`tokenizer.json` or `tokenizer.zip`), and supports both **single-text** and **batch CSV** predictions.

---

** Streamlit Link : **
https://harish-1501-neural-network-final-project-app-3injs6.streamlit.app/


## 🧪 How to Use

### Single-text prediction
1. Paste/enter your text in the left panel.
2. Click **Predict**.
3. You’ll see:
   - **probability** (model’s sigmoid output),
   - **label** determined by your **threshold** (default 0.50; adjustable in sidebar).

### Batch prediction (CSV)
1. Upload a CSV with at least one **text column**.
2. Pick the column from the dropdown.
3. Click **Run batch predictions** to get a preview and a **Download** button.

---

## ⚙️ Tuning the Decision Threshold

If you previously computed a best threshold (for example, maximizing F1 on validation), enter that value in the sidebar.  
- `probability >= threshold` → **positive** (default label name “positive”)  
- otherwise → **negative**

You can also rename the positive/negative labels in the sidebar settings.

---

## ✨ Features

- **Single-text prediction** with probability bar and label.
- **Batch CSV prediction** with preview + downloadable results.
- **Exact training-time tokenizer** (JSON or inside ZIP) applied at inference.
- **Precise threshold control** (number input), with optional **invert probability (1 − p)** toggle.
- **Keras 3–aware loading** (uses `keras.saving.load_model`, falls back to `tf.keras` if needed).
- **Runtime debug panel** showing raw sigmoid and used probability for transparency.

---

## 🧠 Model

- **Architecture:** Embedding (vocab=20,000, dim=128, `mask_zero=True`) → SpatialDropout1D(0.2) → **Bidirectional LSTM(64, return_sequences=True)** → GlobalMaxPooling1D → Dense(64, ReLU) → Dropout(0.3) → Dense(1, **sigmoid**).
- **Input:** int32 token IDs, **shape `[batch, 512]`**, padded/truncated with `0`s (post).
- **Output:** float32 **sigmoid** `P(class=1)` (probability of the trained positive class).
- **Validation/held-out metrics (provided by user):**
  - **best_threshold:** `0.0487249419`
  - **best_val_f1:** `0.9514563107`
  - **ROC AUC:** `0.8839164895`
  - **PR AUC:** `0.9847274865`
  - **Support (test):** pos=`739`, neg=`84`

> ⚠️ Be sure what **class 1** meant in your training (e.g., *toxic*). If your UI calls the opposite class “positive”, either rename labels or use the **Invert probability (1 − p)** toggle.

---

## 🗂️ Repository Structure

```
.
├── app.py                  # Streamlit app
├── requirements.txt        # Python dependencies
├── models/
│   ├── best.keras          # Trained Keras model (Keras 3 format)
│   └── tokenizer.json      # Or tokenizer.zip containing tokenizer.json
├── data/                   # (optional) datasets, samples, exports
└── README.md               # This file
```

> Large model files? Use **Git LFS** or a startup download step.

---

## 🔢 Decision Thresholds

- If **class 1 = toxic** and your UI “Positive label” = **toxic** → set threshold to **`0.0487249419`**.
- If you want “Positive label” = **non-toxic** instead, turn on **Invert probability (1 − p)** and use the **complement threshold**: `1 − 0.0487249419 = 0.9512750581`.

You can change label names and threshold in the app’s **sidebar**.

---

## 💾 Data (YouTube IDs → Comments)

This project’s comments were extracted **from YouTube video IDs** using the **YouTube Data API**. A typical pipeline:

1. **Prepare video IDs**: a CSV or text file of `video_id`s.
2. **Fetch comments** with the `commentThreads.list` endpoint (and optionally `comments.list` for replies).
3. **Store** raw fields such as:
   - `video_id`, `comment_id`, `textOriginal`, `likeCount`, `publishedAt`,
   - `authorDisplayName`, `parentId` (if reply), `replyCount` (thread level).
4. **Clean text** (strip markup, handle emojis, normalize whitespace).
5. **Split** into train/val/test and apply **your tokenizer** for model training.

**Environment variables** commonly used:
```
YOUTUBE_API_KEY=<your_api_key>
```

**Notes & compliance**:
- Respect YouTube’s **Terms of Service**, data retention, and user privacy.
- Observe **API quota limits**; paginate on `nextPageToken`.
- Attribute responsibly and avoid redistributing personal data.

---

## 🧰 Setup

### 1) Python Environment
We recommend Python **3.9–3.11**.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins modern **TensorFlow (≥2.17)** and **Keras (≥3.3)** so models saved with **Keras 3** load correctly.

### 2) Files Required
Place your artifacts as follows:

```
models/
  best.keras
  tokenizer.json        # or tokenizer.zip containing tokenizer.json
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```
Open the URL Streamlit prints (usually http://localhost:8501).

---

## ☁️ Deploy (Streamlit Community Cloud)

1. Push repo to GitHub with the structure above.
2. Go to https://share.streamlit.io and connect your repo.
3. Set **Main file path** to `app.py` and deploy.

> For big model files, use **Git LFS** or load from cloud storage on startup.

---

## 📄 License

MIT (or your preferred license).

---

## 🙌 Acknowledgments

- YouTube Data API for data access.
- Keras / TensorFlow and Streamlit for the ML and UI stack
- Neural Network Course and our guide Mohammad Saiful Islam (https://www.linkedin.com/in/mohammadsaifulislam/)