import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load model & data
vectorizer = joblib.load("model/vectorizer.pkl")
tfidf_matrix = joblib.load("model/tfidf_matrix.pkl")
df = pd.read_csv("data/processed_lyrics.csv")

def predict_song(lyrics, top_n=3):
    lyrics = lyrics.lower()
    lyrics_vec = vectorizer.transform([lyrics])
    similarity = cosine_similarity(lyrics_vec, tfidf_matrix).flatten()
    indices = similarity.argsort()[-top_n:][::-1]
    return df.iloc[indices][['song','artist']]

# ---------------- Streamlit UI ---------------- #

st.title("🎵 Spotify Lyric Search Engine")
st.write("Paste song lyrics, and I will identify the song & artist 💙")

user_input = st.text_area("Enter lyrics snippet here:", height=150)

if st.button("Search"):
    if user_input.strip():
        results = predict_song(user_input)
        st.subheader("Top Matches:")
        for i,row in results.iterrows():
            st.write(f"**{row['song']}** — *{row['artist']}*")
    else:
        st.warning("Please enter some lyrics.")
