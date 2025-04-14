from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import re
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Load model and vectorizer
with open('stress_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()

# Average confidence used in model training
avg_confidence = 0.7  # Update this with actual average used if different

# Cleaning and preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_text(text):
    return ' '.join(word for word in text.split() if word not in stop_words)

def extract_features(text):
    words = nltk.word_tokenize(text)
    num_words = len(words)
    num_positive = sum(1 for word in words if sid.polarity_scores(word)['compound'] > 0.05)
    num_negative = sum(1 for word in words if sid.polarity_scores(word)['compound'] < -0.05)
    return [num_words, num_positive, num_negative]

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('input', '')

    cleaned = clean_text(user_input)
    processed = preprocess_text(cleaned)
    tfidf_input = tfidf.transform([processed]).toarray()

    # Feature engineering
    num_words, num_positive, num_negative = extract_features(processed)
    extra_features = np.array([[avg_confidence, num_words, num_positive, num_negative]])

    combined_input = np.concatenate((tfidf_input, extra_features), axis=1)

    prediction = model.predict(combined_input)[0]
    return jsonify({'prediction': f"Predicted Stress Level (1-10): {prediction:.2f}"})


if __name__ == '__main__':
      
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)