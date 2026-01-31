import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocessing import preprocess_for_model

model = load_model("sentiment_lstm.keras")

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

user_input = st.text_area("Movie Review:")

if st.button("Classify"):
    padded = preprocess_for_model(user_input)
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    st.subheader(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction:.4f}")



