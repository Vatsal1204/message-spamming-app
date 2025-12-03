# app.py
import streamlit as st
import pickle
import re
from datetime import datetime

# Page config
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📨", layout="centered")

# ---------- Loading helper (safe) ----------
@st.cache_resource
def cached_load():
    """
    Return (vectorizer, model, error_message)
    If loading was successful -> error_message is None.
    If something failed -> vectorizer and model are None and error_message contains the error string.
    """
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vec = pickle.load(f)
    except Exception as e:
        return None, None, f"Failed to load 'tfidf_vectorizer.pkl': {e}"

    try:
        with open("spam_model.pkl", "rb") as f:
            mdl = pickle.load(f)
    except Exception as e:
        return None, None, f"Failed to load 'spam_model.pkl': {e}"

    return vec, mdl, None

# ---------- Text cleaning & prediction ----------
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def predict_message(text: str, vectorizer, model):
    """
    Returns (label_str, confidence_float_or_None)
    label_str is 'spam' or 'ham'
    confidence_float_or_None is between 0.0 and 1.0 if available
    """
    x = vectorizer.transform([clean_text(text)])
    pred = model.predict(x)[0]

    # normalize label types (if model used numeric labels)
    if isinstance(pred, (int, float)):
        label = "spam" if int(pred) == 1 else "ham"
    else:
        # sometimes label strings might be 'spam'/'ham' or 'ham'/'spam'
        label = str(pred).lower()
        if label not in ("spam", "ham"):
            # fallback: interpret '1' or '0' strings
            if label.isdigit():
                label = "spam" if int(label) == 1 else "ham"
            else:
                label = "spam" if label in ("true", "yes") else "ham"

    confidence = None
    try:
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(x).max())
    except Exception:
        # if predict_proba fails for any reason, keep None
        confidence = None

    return label, confidence

# ---------- Load artifacts ----------
vectorizer, model, load_error = cached_load()

# ---------- UI ----------
st.markdown("<h1 style='text-align:center;margin-bottom:6px'>📨 SMS Spam Classifier</h1>", unsafe_allow_html=True)

# If loading failed, show the error and stop further execution (no crash).
if load_error:
    st.error("Model files could not be loaded. App cannot run without them.")
    st.caption(load_error)
    st.info("Make sure both `tfidf_vectorizer.pkl` and `spam_model.pkl` are in the same folder as this file.")
    st.stop()

# Input area
message = st.text_area("", height=160, placeholder="Type or paste SMS here...")

# Buttons
col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("Predict")
with col2:
    clear_btn = st.button("Clear")

# Clear action
if clear_btn:
    st.session_state.pop("latest", None)
    st.experimental_rerun()

# Predict action
if predict_btn:
    if not message or not message.strip():
        st.error("Please enter a message.")
    else:
        label, score = predict_message(message, vectorizer, model)
        result = {
            "label": label,
            "score": score,
            "message": message,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        # store for display
        st.session_state.latest = result
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.insert(0, result)
        st.session_state.history = st.session_state.history[:10]

# Show result
if "latest" in st.session_state:
    latest = st.session_state.latest
    st.markdown("---")
    if latest["label"] == "spam":
        st.markdown(
            "<div style='padding:15px;border-radius:10px;background:#fff0f0;border:1px solid #ffd6d6'>"
            "<h3 style='color:#d11a2a;margin:0;'>🚫 SPAM</h3></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='padding:15px;border-radius:10px;background:#e7fff4;border:1px solid #a8e8c9'>"
            "<h3 style='color:#0f8a54;margin:0;'>✅ NOT SPAM</h3></div>",
            unsafe_allow_html=True,
        )
    if latest["score"] is not None:
        st.write(f"**Confidence:** {latest['score'] * 100:.2f}%")
    st.caption(f"Predicted at {latest['time']}")

# Recent predictions
st.markdown("---")
st.subheader("Recent predictions")
if "history" in st.session_state and st.session_state.history:
    for entry in st.session_state.history:
        s = f"{entry['label'].upper()}"
        if entry["score"] is not None:
            s += f" — {entry['score']*100:.1f}%"
        st.write(s)
        st.write(entry["message"])
        st.markdown("---")
else:
    st.info("No predictions yet.")
