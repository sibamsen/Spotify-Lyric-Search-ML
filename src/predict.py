import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load saved data
vectorizer = joblib.load("../model/vectorizer.pkl")
tfidf_matrix = joblib.load("../model/tfidf_matrix.pkl")
df = pd.read_csv("../data/processed_lyrics.csv")

def predict_song(lyrics, top_n=3):
    lyrics = lyrics.lower()
    lyrics_vec = vectorizer.transform([lyrics])

    similarity = cosine_similarity(lyrics_vec, tfidf_matrix).flatten()
    indices = similarity.argsort()[-top_n:][::-1]

    results = df.iloc[indices][['song','artist']]
    return results
