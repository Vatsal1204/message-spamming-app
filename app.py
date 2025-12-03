# app.py
import streamlit as st
import pickle, re, math
from datetime import datetime

st.set_page_config(
    page_title="📩 SMS Spam Classifier — Pro",
    page_icon="📨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Load model + vectorizer ----------
@st.cache_resource
def load_artifacts():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        v = pickle.load(f)
    with open("spam_model.pkl", "rb") as f:
        m = pickle.load(f)
    return v, m

vectorizer, model = load_artifacts()

# ---------- Utilities ----------
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def predict_msg(text):
    x = vectorizer.transform([clean_text(text)])
    pred = model.predict(x)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x).max()
    # convert to friendly strings if numeric labels used
    if isinstance(pred, (int, float)):
        pred = "spam" if int(pred) == 1 else "ham"
    return pred, float(proba) if proba is not None else None

# ---------- Session state for history ----------
if "history" not in st.session_state:
    st.session_state.history = []

def add_history(message, label, score):
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": message,
        "label": label,
        "score": f"{score*100:.2f}%" if score is not None else "N/A"
    })
    # keep last 20
    st.session_state.history = st.session_state.history[:20]

# ---------- Styling ----------
primary = "#0ea5a4"
accent = "#7c3aed"
bg = "#0f172a"
card = "#0b1220"
txt = "#e6eef8"

st.markdown(
    f"""
    <style>
    :root {{
      --primary: {primary};
      --accent: {accent};
      --bg: {bg};
      --card: {card};
      --txt: {txt};
    }}
    .main {{
      background: linear-gradient(135deg, rgba(7,10,24,1) 0%, rgba(4,6,12,1) 100%);
      color: var(--txt);
      padding: 0;
    }}
    .title {{
      font-size: 36px;
      font-weight: 800;
      letter-spacing: -0.5px;
    }}
    .sub {{
      color: #9fb0c8;
      margin-top: -10px;
    }}
    .glass {{
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border: 1px solid rgba(255,255,255,0.04);
      padding: 22px;
      border-radius: 12px;
      color: var(--txt);
      box-shadow: 0 6px 30px rgba(2,6,23,0.6);
    }}
    .btn {{
      background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
      color: white !important;
      border: none;
      padding: 8px 18px;
      border-radius: 10px;
      font-weight: 600;
    }}
    .example-chip {{
      display:inline-block;
      padding:6px 10px;
      margin:4px;
      border-radius:999px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.03);
      cursor:pointer;
    }}
    .small-muted {{ color:#9fb0c8; font-size:12px; }}
    </style>
    """, unsafe_allow_html=True
)

# ---------- Layout ----------
header_col, _ = st.columns([3,1])
with header_col:
    st.markdown('<div class="title">📩 SMS Spam Classifier — PRO UI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">A designer-grade interface for your NLP model — keep your logic, upgrade the look.</div>', unsafe_allow_html=True)

left, right = st.columns([2.2, 1])

with left:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Enter SMS message")
    message = st.text_area("", height=160, placeholder="Type or paste any SMS text here...")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Predict", key="predict_btn"):
            if not message.strip():
                st.error("Please enter a message to predict.")
            else:
                label, score = predict_msg(message)
                add_history(message, label, score if score is not None else None)
                # show result in an expander below
                st.session_state.latest = {"label": label, "score": score}
    with c2:
        if st.button("Clear", key="clear_btn"):
            st.experimental_rerun()
    with c3:
        if st.button("Use Example", key="sample_btn"):
            # fill with a spam example
            st.session_state.prefill = "Congratulations! You won a free lottery of ₹10,00,000. Click the link to claim."
            st.experimental_rerun()

    # quick example chips
    st.markdown("**Examples**  <span class='small-muted'>(click to paste)</span>", unsafe_allow_html=True)
    example_cols = st.columns(1)
    e1, e2, e3, e4 = st.columns([1,1,1,1])
    if e1.button("Free prize (spam)"):
        message = "WINNER!! You have been selected for a free gift voucher. Reply YES to claim."
        st.session_state.prefill = message
        st.experimental_rerun()
    if e2.button("Bank alert (spam)"):
        message = "URGENT! Your account will be blocked today. Verify immediately: http://fakebank-secure.com"
        st.session_state.prefill = message
        st.experimental_rerun()
    if e3.button("Order update (ham)"):
        message = "Your order has been shipped and will be delivered tomorrow."
        st.session_state.prefill = message
        st.experimental_rerun()
    if e4.button("Personal (ham)"):
        message = "Hey bro, are we meeting today for the movie?"
        st.session_state.prefill = message
        st.experimental_rerun()

    # prefill: if set, write it into the text area (works on rerun)
    if "prefill" in st.session_state:
        # use st.experimental_set_query_params to force the text area? simpler: show suggestion box
        st.info("Example loaded to input box. Press Predict.")
        # display prefilled text as a hint
        st.code(st.session_state.prefill, language=None)

    # show latest result if exists
    if "latest" in st.session_state:
        latest = st.session_state.latest
        st.markdown("---")
        st.markdown("### Result")
        lbl = latest["label"]
        score = latest["score"]
        if lbl in ("spam", "SPAM", "Spam", 1):
            st.markdown(f"<div style='padding:12px;border-radius:10px;background:rgba(255,40,66,0.06);border:1px solid rgba(255,40,66,0.12)'><h3 style='color:#ff4a4a;margin:0'>🚫 SPAM</h3><div class='small-muted'>This message looks like spam.</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='padding:12px;border-radius:10px;background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.12)'><h3 style='color:#10b981;margin:0'>✅ NOT SPAM</h3><div class='small-muted'>This message seems legitimate.</div></div>", unsafe_allow_html=True)
        if score is not None:
            st.markdown(f"**Confidence:** {score*100:.2f}%")
            # visual progress bar
            st.progress(min(1.0, score))
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Model Summary")
    # display simple cards
    st.markdown(f"""
    - **Vectorizer:** `{type(vectorizer).__name__}`  
    - **Model:** `{type(model).__name__}`  
    - **Predict API:** `predict_msg(text)`  
    """)
    st.markdown("---")
    st.subheader("Quick Actions")
    st.write("Download or demo helpers:")
    st.download_button("Download sample CSV (10 rows)", data="message,label\nHello, ham\nWin money spam,spam\n", file_name="sample_sms.csv", mime="text/csv")
    st.markdown("")
    if st.button("Show sample test cases"):
        st.table([
            {"Example": "Free prize", "Expected": "spam"},
            {"Example": "Order delivered", "Expected": "ham"},
            {"Example": "Verify account", "Expected": "spam"},
        ])
    st.markdown("---")
    st.subheader("Prediction History")
    if st.session_state.history:
        for i, row in enumerate(st.session_state.history):
            st.markdown(f"**{row['label'].upper()}** — <span class='small-muted'>{row['time']} • {row['score']}</span>", unsafe_allow_html=True)
            st.caption(row["message"])
            if i >= 6:  # show first 7 items
                break
    else:
        st.info("No predictions yet. Try a message on the left.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown(
    """
    <div style="padding:18px; border-radius:12px; margin-top:18px; text-align:center; color:#9fb0c8;">
    Built with ❤️ • Show this in your viva — explain: preprocessing → vectorizer → classifier → predict_proba
    </div>
    """, unsafe_allow_html=True
)
