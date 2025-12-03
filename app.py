# app.py
import streamlit as st
import pickle, re
from datetime import datetime

# Page config
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📨", layout="centered")

# ---- Helpers ----
def load_artifacts():
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vec = pickle.load(f)
        with open("spam_model.pkl", "rb") as f:
            mdl = pickle.load(f)
        return vec, mdl, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def cached_load():
    return load_artifacts()

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def predict(text: str, vectorizer, model):
    x = vectorizer.transform([clean_text(text)])
    pred = model.predict(x)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x).max()
    # normalize numeric labels
    if isinstance(pred, (int, float)):
        pred = "spam" if int(pred) == 1 else "ham"
    return str(pred), float(prob) if prob is not None else None

# ---- Load model & vectorizer ----
vectorizer, model, load_error = cached_load()

# ---- Header ----
st.title("📨 SMS Spam Classifier")
st.write("Simple, clean interface. Paste an SMS and press **Predict**.")

# If loading failed, show error and stop
if load_error:
    st.error("Model files could not be loaded. Make sure `spam_model.pkl` and `tfidf_vectorizer.pkl` are in the same folder as this file.")
    st.code(load_error)
    st.stop()

# ---- Layout: main input and result ----
message = st.text_area("Enter SMS message", height=160, placeholder="Type or paste the SMS here...")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Predict"):
        if not message.strip():
            st.error("Please enter a message to predict.")
        else:
            label, score = predict(message, vectorizer, model)
            st.session_state.latest = {"label": label, "score": score, "message": message, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            # store history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(0, st.session_state.latest)
            # limit history
            st.session_state.history = st.session_state.history[:10]
with col2:
    if st.button("Clear"):
        st.session_state.pop("latest", None)
        st.experimental_rerun()

# Show result if available
if "latest" in st.session_state:
    latest = st.session_state.latest
    st.markdown("---")
    if latest["label"].lower() in ("spam", "1"):
        st.markdown('<div style="padding:14px;border-radius:10px;background:#fff0f0;border:1px solid #ffd6d6"><h3 style="color:#b91c1c;margin:0">🚫 SPAM</h3><div style="color:#6b7280">This message is classified as spam.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:14px;border-radius:10px;background:#f0fffb;border:1px solid #c7f3e8"><h3 style="color:#047857;margin:0">✅ NOT SPAM</h3><div style="color:#6b7280">This message looks legitimate.</div></div>', unsafe_allow_html=True)
    if latest["score"] is not None:
        st.write(f"**Confidence:** {latest['score']*100:.2f}%")
    st.caption(f"Predicted at {latest['time']}")

# ---- History (small) ----
st.markdown("----")
st.subheader("Recent predictions")
if "history" in st.session_state and st.session_state.history:
    for entry in st.session_state.history:
        label = entry["label"].upper()
        time = entry["time"]
        score = f"{entry['score']*100:.1f}%" if entry["score"] is not None else "N/A"
        st.write(f"**{label}** — {time} — {score}")
        st.write(entry["message"])
        st.markdown("---")
else:
    st.info("No predictions yet. Paste a message and press Predict.")

# ---- Sidebar: model info ----
with st.sidebar:
    st.header("Model info")
    st.write(f"- Vectorizer: `{type(vectorizer).__name__}`")
    st.write(f"- Model: `{type(model).__name__}`")
    st.write("- Keep `tfidf_vectorizer.pkl` and `spam_model.pkl` in same folder as this file.")
    st.markdown("---")
    st.write("Tip: For deployment, upload this repository to GitHub and use Streamlit Cloud.")

# End
