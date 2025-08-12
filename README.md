# Text Classification Web App (Streamlit + Keras)

A simple Streamlit UI that loads your saved Keras model (`best.keras`) and tokenizer (`tokenizer.json` or `tokenizer.zip`). 
It accepts raw text, applies the exact tokenizer you trained with, pads/truncates to 512 tokens, and returns the model’s sigmoid probability for the positive class.

---

## 🧱 Project Structure

```
your-project/
  models/
    best.keras
    tokenizer.json        # or tokenizer.zip (must contain tokenizer.json)
  app.py                  # from the canvas
  requirements.txt        # this file
  README.md               # this file
```

> **Model I/O (from your uploaded model):**  
> • **Input:** int32 token IDs, shape `[batch, 512]`, padded with `0`s (mask_zero).  
> • **Output:** float32 sigmoid, shape `[batch, 1]` (probability of positive class).

---

## 💾 Installation

1) (Optional) Create a fresh virtual environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** We pin `numpy<2.0` because TensorFlow 2.15 is not compatible with NumPy 2.x.

---

## ▶️ Run Locally

Make sure `best.keras` and your tokenizer are inside `models/` as shown above, then run:
```bash
streamlit run app.py
```

Open the URL that Streamlit prints (usually http://localhost:8501).

---

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

## 🚀 Deploy on Streamlit Community Cloud

1. Push your repository to GitHub with this layout:
   - `app.py`
   - `requirements.txt`
   - `models/best.keras`
   - `models/tokenizer.json` (or `models/tokenizer.zip`)

2. Go to **https://share.streamlit.io** and connect your repo.
3. Set **Main file path** to `app.py` and deploy.

> Tip: Large model files can exceed GitHub’s free limits. If needed, use Git LFS or host the model on cloud storage and download it at app startup.

---

## ❗ Troubleshooting

- **“Model not found”** — Ensure `models/best.keras` exists relative to `app.py`.
- **“Tokenizer not found”** — Ensure `models/tokenizer.json` (or `models/tokenizer.zip` containing a `tokenizer.json`) exists.
- **Import/ABI errors with TensorFlow** — Use Python 3.9–3.11 with `tensorflow==2.15.0`. On Apple Silicon, the CPU wheel is usually fine for this app.
- **Blank or NaN probabilities** — Check that the tokenizer used at inference is the exact one from training (same vocab and preprocessing).

---

## 🔒 Notes

- This app is **UI-only**. If you need a REST API, we can add a small FastAPI or Flask service with the same preprocessing and model.
- Text longer than **512 tokens** will be truncated on the right (post-truncation). Shorter text is padded with zeros.

---

## 📜 License

MIT (or your preferred license).
