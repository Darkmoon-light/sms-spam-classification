
import pickle
from pathlib import Path

import streamlit as st

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="SMS Spam Shield | AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "spam_model.pkl"


# ---------------- LOAD YOUR EXISTING MODEL ----------------
@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as file:
        vectorizer, model = pickle.load(file)
    return vectorizer, model


# ---------------- DARK BLUE DESIGN ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #07111f 0%, #0b1730 55%, #101e3d 100%);
        color: #edf4ff;
    }

    [data-testid="stSidebar"] {
        background: #081426;
        border-right: 1px solid #203453;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 30px;
        border: 1px solid #294a7a;
        border-radius: 20px;
        background: linear-gradient(120deg, #102b52, #174ea6);
        margin-bottom: 24px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.22);
    }

    .hero h1 {
        color: #ffffff;
        font-size: 2.4rem;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #d4e5ff;
        font-size: 1.05rem;
        margin: 0;
    }

    .panel {
        padding: 22px;
        border: 1px solid #263b5c;
        border-radius: 16px;
        background: rgba(15, 31, 56, 0.88);
        margin-bottom: 18px;
    }

    .muted {
        color: #a8bad4;
        font-size: 0.92rem;
    }

    div.stButton > button[kind="primary"] {
        background: #2563eb;
        border: 1px solid #3b82f6;
        color: white;
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #1d4ed8;
        border-color: #60a5fa;
    }

    textarea {
        border-radius: 12px !important;
    }

    div[data-testid="stMetric"] {
        background: #101f38;
        border: 1px solid #263b5c;
        padding: 16px;
        border-radius: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🛡️ Spam Shield")
    st.caption("AI / NLP PROJECT")
    st.divider()

    st.markdown("### Dashboard")
    st.write("📩 SMS classification")
    st.write("🧠 TF-IDF + Naive Bayes")

    st.divider()
    st.markdown("### Model information")
    st.write("**Task:** Text classification")
    st.write("**Classes:** Spam / Ham")
    st.write("**Status:** Model loaded when available")

    st.divider()
    st.caption("Educational demonstration. Predictions may be incorrect.")


# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ SMS Spam Shield</h1>
        <p>Intelligent message screening powered by Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Message Analyzer")
st.markdown(
    '<p class="muted">Enter an SMS message below to predict whether it is '
    'spam or a legitimate message (HAM).</p>',
    unsafe_allow_html=True,
)


# ---------------- MODEL LOADING ----------------
try:
    vectorizer, model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Please put spam_model.pkl in the same "
        "folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error("The saved model could not be loaded.")
    st.caption(f"Technical details: {exc}")
    st.stop()


# ---------------- MESSAGE INPUT ----------------
with st.container(border=True):
    st.markdown("### 📩 Enter your message")

    message = st.text_area(
        "SMS text",
        placeholder="Example: Hey, are we meeting at college tomorrow?",
        height=160,
        max_chars=5000,
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1.4, 1, 3])

    with col1:
        check_clicked = st.button(
            "🔍 Analyze Message",
            type="primary",
            use_container_width=True,
        )

    with col2:
        clear_clicked = st.button(
            "Clear",
            use_container_width=True,
        )

    with col3:
        st.caption("Your text is processed by your saved local model.")

if clear_clicked:
    st.rerun()


# ---------------- PREDICTION RESULTS ----------------
if check_clicked:
    if not message.strip():
        st.warning("Please enter a message before analyzing.")
    else:
        features = vectorizer.transform([message])
        prediction = model.predict(features)[0]
        prediction_text = str(prediction).strip().lower()

        st.divider()
        st.markdown("## Analysis Result")

        if prediction_text == "spam":
            st.error("🚨 SPAM DETECTED")
            st.markdown(
                "The model classifies this message as **SPAM**. "
                "Treat unexpected offers, payment requests, and links cautiously."
            )
        else:
            st.success("✅ LIKELY LEGITIMATE (HAM)")
            st.markdown(
                "The model classifies this message as **HAM**. "
                "This does not guarantee that the message is safe."
            )

        left, right = st.columns(2)

        with left:
            st.metric("Predicted category", prediction_text.upper())

        with right:
            st.metric("Characters analyzed", len(message))

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            classes = list(model.classes_)

            st.markdown("### Model confidence estimates")
            st.caption(
                "These are model probability estimates, not a guarantee "
                "that the prediction is correct."
            )

            for label, probability in zip(classes, probabilities):
                st.write(f"**{str(label).upper()}** — {probability * 100:.1f}%")
                st.progress(float(probability))


# ---------------- HOW IT WORKS ----------------
st.divider()

with st.expander("🧠 How does this AI project work?"):
    st.markdown(
        """
        1. **Input:** A user enters an SMS message.
        2. **TF-IDF:** The text is converted into numerical features.
        3. **Naive Bayes:** The trained classifier predicts spam or ham.
        4. **Output:** The website displays the predicted category.

        The model was trained separately. This app loads the saved
        vectorizer and classifier from `spam_model.pkl`.
        """
    )

st.markdown(
    """
    <p style="text-align:center; color:#91a4c2; padding-top:25px;">
    SMS Spam Shield · AI Applications Lab · TF-IDF + Multinomial Naive Bayes
    </p>
    """,
    unsafe_allow_html=True,
)
