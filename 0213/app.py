import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Enable MPS
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load trained model and move it to MPS
model_path = "./trained_lora_model"
model = BertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define a prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Move input tensors to MPS
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits

    prediction = torch.argmax(logits, dim=-1).item()
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive", 3: "Irrelevant"}
    return sentiment_map[prediction]

# Streamlit Web App
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet below to analyze its sentiment:")

# User input
user_input = st.text_input("Enter tweet:")

if user_input:
    result = predict(user_input)
    st.write(f"Sentiment: {result}")
