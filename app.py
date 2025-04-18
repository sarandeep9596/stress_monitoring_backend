from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load pre-trained model and TF-IDF
model = joblib.load("stress_model.pkl")       # Replace with your model file
tfidf = joblib.load("tfidf_vectorizer.pkl")       # Replace with your TF-IDF vectorizer

# For demo: avg_confidence can be fixed or calculated if you want
avg_confidence = 0.85

# -----------------------
# Utilities for Text Prep
# -----------------------

def clean_text(text):
    return text.lower().strip()

def preprocess_text(text):
    return text  # Extend this if needed

def extract_features(text):
    words = text.split()
    num_words = len(words)
    # Dummy values - Replace with actual sentiment analysis if available
    num_positive = sum(word in ["happy", "good", "great"] for word in words)
    num_negative = sum(word in ["sad", "bad", "stress"] for word in words)
    return num_words, num_positive, num_negative

# -----------------------
# Request Body Schema
# -----------------------

class TextInput(BaseModel):
    input: str

# -----------------------
# Prediction Route
# -----------------------

@app.post("/predict")
async def predict_stress(data: TextInput):
    user_input = data.input

    cleaned = clean_text(user_input)
    processed = preprocess_text(cleaned)
    tfidf_input = tfidf.transform([processed]).toarray()

    # Feature engineering
    num_words, num_positive, num_negative = extract_features(processed)
    extra_features = np.array([[avg_confidence, num_words, num_positive, num_negative]])

    # Combine features
    combined_input = np.concatenate((tfidf_input, extra_features), axis=1)

    # Predict stress level
    prediction = model.predict(combined_input)[0]

    return {"prediction": f"Predicted Stress Level (1-10): {prediction:.2f}"}
