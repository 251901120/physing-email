from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "phishing_model.joblib"

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="centered")
st.title("🛡️ Phishing Email Detector")
st.write("Paste an email message to check whether its content resembles phishing or a legitimate email.")

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet. Run `python train_model.py` in the project folder first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

email_text = st.text_area(
    "Email subject and body",
    height=220,
    placeholder="Paste the complete email here, including any URLs..."
)

if st.button("Analyze Email", type="primary", use_container_width=True):
    if not email_text.strip():
        st.error("Please paste an email before analyzing.")
    else:
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)])

        if prediction == "Phishing":
            st.error("⚠️ Prediction: PHISHING")
            st.write("Do not click links or share passwords, OTPs, or banking information until you verify the sender independently.")
        else:
            st.success("✅ Prediction: SAFE (model assessment)")
            st.write("The model did not detect enough phishing-like patterns. This is not a guarantee that the email is genuine.")

        st.metric("Model confidence", f"{confidence:.1%}")
        st.caption("Confidence is the model's estimated probability, not a guarantee of correctness.")

st.divider()
st.subheader("Important")
st.write("This is an educational machine-learning prototype. Check the sender address, domain, and message context manually. Never rely on this model alone for security decisions.")
