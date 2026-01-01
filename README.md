# 🎵 Spotify Lyric Search ML App

---
A machine learning based lyric identification system that predicts **Song + Artist** from any lyrics snippet.  
Paste a line of lyrics → instantly get the most similar song using **TF-IDF + Cosine Similarity**.

---

## 🚀 Tech Stack

- **Python**
- **TF-IDF Vectorization**
- **Cosine Similarity (Song Retrieval)**
- **Streamlit Web App**
- **Spotify 50k+ Song Lyrics Dataset**
- **Deployed using Streamlit Cloud**

---

## 📂 Project Structure

Spotify-Lyric-Search-ML/
│── data/
│ └── processed_lyrics.csv
│── model/
│ ├── vectorizer.pkl
│ └── tfidf_matrix.pkl
│── deployment/
│ └── app.py
│── src/
│ └── predict.py
│── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_model_training.ipynb
│ └── 03_prediction_testing.ipynb
│── requirements.txt
│── README.md


---

## 🧠 How It Works

1. Clean lyrics (lowercase, remove stopwords, punctuation)
2. Convert lyrics into vectors using **TF-IDF**
3. Compare user input lyrics with the entire dataset
4. Return **Top most similar songs + artists**

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run deployment/app.py

---

🌐 Live Deployment

https://your-app-name.streamlit.app

✨ Features

🔎 Enter lyrics and get matching song instantly

⚡ Fast retrieval model

🎼 Works with incomplete lyrics/snippets

🌍 Web UI using Streamlit

🧠 No heavy GPU required


🔥 Future Improvements

Use BERT/Sentence Transformers for better semantic match

Add song preview/audio link

Artist-based filtering

API endpoint support

📌 Author

Sibam Sen 💙
Contributions & suggestions are welcome!