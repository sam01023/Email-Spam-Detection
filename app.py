import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# ---------- Page Config ----------
st.set_page_config(
    page_title="Email Spam Detector",
    layout="centered",
)

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f7f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 40px;
        font-weight: 700;
        color: #333;
        text-align: center;
        margin-top: 20px;
    }
    .subtitle {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction {
        font-size: 24px;
        text-align: center;
        margin-top: 30px;
    }

    /* Dark mode text area */
    textarea {
        background-color: #1e1e1e !important;
        color: #f1f1f1 !important;
        font-size: 16px !important;
        border: 1px solid #444 !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown('<div class="title">📧 BERT Spam Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste your email content and get instant results</div>', unsafe_allow_html=True)

# ---------- Load model/tokenizer ----------
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert_spam_model")
    model = BertForSequenceClassification.from_pretrained("bert_spam_model")
    return tokenizer, model

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---------- Input ----------
email = st.text_area("Paste email text here", height=200)

# ---------- Prediction ----------
if st.button("🚀 Detect Spam"):
    if not email.strip():
        st.warning("Please enter some email text.")
    else:
        inputs = tokenizer(email, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        label = "🛑 SPAM" if pred == 1 else "✅ HAM"
        color = "#ff4d4d" if pred == 1 else "#28a745"

        st.markdown(
            f"""
            <div class="prediction" style="color:{color};">
                Prediction: <strong>{label}</strong><br>
                Confidence: <strong>{confidence * 100:.2f}%</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color: #888;'>Made with ❤️ by Stark Industries using BERT & Streamlit</p>",
    unsafe_allow_html=True,
)
