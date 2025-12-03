# app.py
import streamlit as st
import pickle
import re
from datetime import datetime

# Page config
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📨", layout="centered")

# ---- Helpers ----
def load_artifacts():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    with open("spam_model.pkl", "rb") as f:
        mdl = pickle.load(f)
    return vec, mdl

@st.cache_resource
def cached_load():
    return load_artifacts()

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def predict(text, vectorizer, model):
    x = vectorizer.transform([clean_text(text)])
    pred = model.predict(x)[0]
    prob = model.predict_proba(x).max() if hasattr(model, "predict_proba") else None
    if isinstance(pred, (int, float)):
        pred = "spam" if int(pred) == 1 else "ham"
    return pred, prob

# ---- Load model & vectorizer ----
vectorizer, model = cached_load()

# ---- Title ----
st.markdown("<h1 style='text-align:center;'>📨 SMS Spam Classifier</h1>", unsafe_allow_html=True)

# ---- Input area ----
message = st.text_area("", height=160, placeholder="Type or paste SMS here...")

# ---- Buttons ----
c1, c2 = st.columns([1, 1])
with c1:
    predict_btn = st.button("Predict")
with c2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.session_state.pop("latest", None)
    st.experimental_rerun()

# ---- Prediction ----
if predict_btn:
    if not message.strip():
        st.error("Please enter a message.")
    else:
        label, score = predict(message, vectorizer, model)
        st.session_state.latest = {
            "label": label,
            "score": score,
            "message": message,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.insert(0, st.session_state.latest)
        st.session_state.history = st.session_state.history[:10]

# ---- Show result ----
if "latest" in st.session_state:
    latest = st.session_state.latest
    st.markdown("---")
    if latest["label"] == "spam":
        st.markdown(
            "<div style='padding:15px;border-radius:10px;background:#ffecec;border:1px solid #ffb3b3;'>"
            "<h3 style='color:#d11a2a;margin:0;'>🚫 SPAM</h3></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='padding:15px;border-radius:10px;background:#e7fff4;border:1px solid #a8e8c9;'>"
            "<h3 style='color:#0f8a54;margin:0;'>✅ NOT SPAM</h3></div>",
            unsafe_allow_html=True,
        )
    if latest["score"] is not None:
        st.write(f"**Confidence:** {latest['score'] * 100:.2f}%")
    st.caption(f"Predicted at {latest['time']}")

# ---- Recent predictions ----
st.markdown("---")
st.subheader("Recent predictions")

if "history" in st.session_state and st.session_state.history:
    for entry in st.session_state.history:
        st.write(f"**{entry['label'].upper()}** — {entry['score'] * 100:.1f}%")
        st.write(entry["message"])
        st.markdown("---")
else:
    st.info("No predictions yet.")
