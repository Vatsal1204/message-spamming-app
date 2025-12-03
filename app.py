# app.py
import streamlit as st
import pickle
import re
from datetime import datetime

# Page config
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📨", layout="centered")

# ---------- Safe load ----------
@st.cache_resource
def cached_load():
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

vectorizer, model, load_error = cached_load()

# ---------- Helpers ----------
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def predict_message(text: str, vectorizer, model):
    x = vectorizer.transform([clean_text(text)])
    pred = model.predict(x)[0]
    # normalise label
    if isinstance(pred, (int, float)):
        label = "spam" if int(pred) == 1 else "ham"
    else:
        label = str(pred).lower()
        if label not in ("spam", "ham"):
            if label.isdigit():
                label = "spam" if int(label) == 1 else "ham"
            else:
                label = "spam" if label in ("true", "yes") else "ham"
    confidence = None
    try:
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(x).max())
    except Exception:
        confidence = None
    return label, confidence

# ---------- Show error if files missing ----------
if load_error:
    st.markdown("<h1 style='text-align:center;'>📨 SMS Spam Classifier</h1>", unsafe_allow_html=True)
    st.error("Model files could not be loaded. App cannot run without them.")
    st.caption(load_error)
    st.info("Place both `tfidf_vectorizer.pkl` and `spam_model.pkl` in the same folder as this app.")
    st.stop()

# ---------- Initialize session state ----------
if "input_message" not in st.session_state:
    st.session_state["input_message"] = ""
if "history" not in st.session_state:
    st.session_state["history"] = []
if "latest" not in st.session_state:
    st.session_state["latest"] = None

# ---------- UI ----------
st.markdown("<h1 style='text-align:center;margin-bottom:6px'>📨 SMS Spam Classifier</h1>", unsafe_allow_html=True)

# bind the text_area to session state so Clear can empty it reliably
st.text_area("", key="input_message", height=160, placeholder="Type or paste SMS here...")

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("Predict")
with col2:
    clear_btn = st.button("Clear")

# Clear action: simply empty the bound session_state value and latest
if clear_btn:
    st.session_state["input_message"] = ""
    st.session_state["latest"] = None

# Predict action
if predict_btn:
    msg = st.session_state.get("input_message", "").strip()
    if not msg:
        st.error("Please enter a message.")
    else:
        label, score = predict_message(msg, vectorizer, model)
        result = {
            "label": label,
            "score": score,
            "message": msg,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.session_state["latest"] = result
        st.session_state["history"].insert(0, result)
        st.session_state["history"] = st.session_state["history"][:10]

# Show result
if st.session_state["latest"]:
    latest = st.session_state["latest"]
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
if st.session_state["history"]:
    for entry in st.session_state["history"]:
        s = f"{entry['label'].upper()}"
        if entry["score"] is not None:
            s += f" — {entry['score']*100:.1f}%"
        st.write(s)
        st.write(entry["message"])
        st.markdown("---")
else:
    st.info("No predictions yet.")
