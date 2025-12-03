# app.py
import streamlit as st
import pickle
import re
from datetime import datetime

# Page config
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📨", layout="centered")

# ---------- Safe load ----------
@st.cache_resource
def load_files():
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    except Exception as e:
        return None, None, f"Error loading tfidf_vectorizer.pkl: {e}"

    try:
        with open("spam_model.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return None, None, f"Error loading spam_model.pkl: {e}"

    return vectorizer, model, None

vectorizer, model, load_error = load_files()

# ---------- Error if model missing ----------
if load_error:
    st.title("📨 SMS Spam Classifier")
    st.error("Model files could not be loaded.")
    st.caption(load_error)
    st.stop()

# ---------- Helpers ----------
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def classify(text):
    x = vectorizer.transform([clean_text(text)])
    pred = model.predict(x)[0]
    # normalize type
    if isinstance(pred, (int, float)):
        pred = "spam" if int(pred) == 1 else "ham"
    else:
        pred = pred.lower()

    conf = None
    if hasattr(model, "predict_proba"):
        try:
            conf = float(model.predict_proba(x).max())
        except:
            conf = None
    return pred, conf

# ---------- UI ----------
st.markdown("<h1 style='text-align:center;'>📨 SMS Spam Classifier</h1>", unsafe_allow_html=True)

message = st.text_area("", height=160, placeholder="Type or paste SMS here...")

predict_btn = st.button("Predict")

# ---------- Predict ----------
if predict_btn:
    if not message.strip():
        st.error("Please enter a message.")
    else:
        label, score = classify(message)

        st.markdown("---")
        if label == "spam":
            st.markdown(
                "<div style='padding:15px;border-radius:10px;background:#ffecec;border:1px solid #ffb3b3;'>"
                "<h3 style='color:#cc0000;margin:0;'>🚫 SPAM</h3></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='padding:15px;border-radius:10px;background:#e7fff4;border:1px solid #a8e8c9;'>"
                "<h3 style='color:#0f8a54;margin:0;'>✅ NOT SPAM</h3></div>",
                unsafe_allow_html=True,
            )

        if score is not None:
            st.write(f"**Confidence:** {score * 100:.2f}%")
