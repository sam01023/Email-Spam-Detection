# 🤖 BERT Email Spam Detector

A fast, fine-tuned, and beautifully designed email spam detector powered by BERT and wrapped in a dark-mode Streamlit app. Because spam deserves a punch to the throat—with science.

---

## 📌 Overview

This project uses a **fine-tuned BERT model** to classify emails as **spam** or **ham** (non-spam).  
It includes:

- Preprocessing of email content  
- Tokenization using HuggingFace `transformers`  
- Training and evaluation with `Trainer API`  
- Aesthetic **Streamlit** frontend with dark-mode input box  
- Git LFS support for heavy model files

---

## 📁 Project Structure

📦BERT-Spam-Detector
├── app.py # Streamlit frontend
├── requirements.txt # Dependencies
├── email.csv # Dataset (from Kaggle)
├── bert_spam_model/ # Fine-tuned BERT model
│ ├── config.json
│ ├── model.safetensors
│ ├── tokenizer_config.json
│ ├── vocab.txt
├── spam_detection.ipynb # Jupyter notebook for training
└── .gitattributes # For Git LFS



---

## 🧠 Model Details

- **Model Base**: `bert-base-uncased`
- **Task**: Binary classification (`ham = 0`, `spam = 1`)
- **Dataset**: [Kaggle Email Spam Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Tokenizer**: HuggingFace Tokenizer
- **Training Framework**: HuggingFace Transformers with PyTorch backend
- **Frontend**: Streamlit with dark theme applied to input area

---

## 💻 How to Run Locally

### ✅ 1. Clone the repo

```bash
git clone https://github.com/sam01023/Email-Spam-Detection.git
cd Email-Spam-Detection
