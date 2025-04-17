import pandas as pd
import numpy as np
import re
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import nltk
from nltk.tokenize import word_tokenize 
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
df = pd.read_csv('Stress.csv')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_comment'] = df['text'].apply(clean_text)

stemmer = PorterStemmer()
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(stemmer.stem(word) for word in text.split() if word not in stop_words)

df['processed_comment'] = df['cleaned_comment'].apply(preprocess_text)

sid = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['processed_comment'].apply(lambda x: sid.polarity_scores(x)['compound'])

def scale_stress(row):
    if row['label'] == 0:
        base_stress = 1 + (1 - row['sentiment_score']) * 4.5
    else:
        base_stress = 5.5 + (1 - row['sentiment_score']) * 4.5
    return base_stress * row['confidence']

df['stress_level_scaled'] = df.apply(scale_stress, axis=1)

def extract_features(text):
    words = nltk.word_tokenize(text)
    num_words = len(words)
    num_positive = sum(1 for word in words if sid.polarity_scores(word)['compound'] > 0.05)
    num_negative = sum(1 for word in words if sid.polarity_scores(word)['compound'] < -0.05)
    return pd.Series([num_words, num_positive, num_negative])

df[['num_words', 'num_positive', 'num_negative']] = df['processed_comment'].apply(extract_features)

tfidf = TfidfVectorizer(max_features=500)
X_tfidf = tfidf.fit_transform(df['processed_comment']).toarray()

X = pd.DataFrame(X_tfidf, columns=tfidf.get_feature_names_out())
X['confidence'] = df['confidence']
X['num_words'] = df['num_words']
X['num_positive'] = df['num_positive']
X['num_negative'] = df['num_negative']

y = df['stress_level_scaled']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Model Trained. Mean Squared Error: {mse:.2f}')

pickle.dump(model, open('stress_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))

def predict_stress_level(text_input):
    cleaned = clean_text(text_input)
    processed = preprocess_text(cleaned)
    features = extract_features(processed)

    tfidf_input = tfidf.transform([processed]).toarray()
    combined = np.concatenate((tfidf_input, [[df['confidence'].mean(), *features]]), axis=1)

    return model.predict(combined)[0]

if __name__ == "__main__":
    user_input = input("Enter a statement about your day: ")
    stress = predict_stress_level(user_input)
    print(f"Predicted Stress Level (1-10): {stress:.2f}")

