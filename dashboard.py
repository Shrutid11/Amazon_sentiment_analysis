import streamlit as st
import joblib
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "model", "tfidf.pkl")

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

st.title("Amazon Review Sentiment Analysis")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

review = st.text_area("Enter your review", key="review_input")

if st.button("Predict"):
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    st.success(f"Sentiment: {prediction}")
