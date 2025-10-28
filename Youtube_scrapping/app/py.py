import streamlit as st
import pandas as pd
from transformers import pipeline
import random

# Load reclassified dataset
data_translated = pd.read_csv("comments_with_sentiment_updt.csv", encoding="utf-8")

# Load pre-trained sentiment model
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual",
        tokenizer="AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual",
        device=-1,
        truncation=True,
        max_length=512
    )

sentiment_model = load_model()

st.set_page_config(page_title="YouTube Comment Sentiment App", page_icon="💬")
st.title("Real-time Sentiment Analysis")
st.write("Realtime analysis with **XLM-Roberta**")

# 1️⃣ Realtime user input
text = st.text_area("Write any comment:", placeholder="E.g. Michael Jackson was amazing!")

if st.button("Analyze Comment"):
    if text.strip() == "":
        st.warning("Please enter a comment.")
    else:
        with st.spinner("Analyzing..."):
            result = sentiment_model(text)
        label = result[0]['label']
        score = result[0]['score']
        if "POS" in label.upper():
            st.success(f"😊 Positive — (score: {score:.2f})")
        elif "NEG" in label.upper():
            st.error(f"😞 Negative — (score: {score:.2f})")
        else:
            st.info(f"😐 Neutral — (score: {score:.2f})")
        st.write("Detailed result:")
        st.json(result)

st.markdown("---")

# 2️⃣ Dataset stats after reclassification
st.subheader("📊 Dataset sentiment distribution after confidence-based reclassification")
st.bar_chart(data_translated['sentiment_updated'].value_counts())

st.markdown("---")



st.caption("Developed by Sevinj 💫 | Model: AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual")
