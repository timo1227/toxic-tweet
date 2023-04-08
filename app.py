import torch
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Title
st.title("Sentiment Analysis")

# Text input
text = st.text_input("Enter your text here")

# Button
if st.button("Predict"):
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits # type: ignore

    # Get the predicted class id
    predicted_class_id = logits.argmax().item()

    # Print if the prediction is positive or negative
    if predicted_class_id == 0:
        st.write("Negative")
    else:
        st.write("Positive")