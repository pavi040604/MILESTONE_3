import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import vosk
import json
import pyaudio
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load product and objection data
@st.cache_resource
def load_data():
    product_data = pd.read_csv("PRODUCT_DATA_PATH")
    objections_data = pd.read_csv("OBJECTIONS_DATA_PATH")
    return product_data, objections_data

product_data, objections_data = load_data()

product_descriptions = product_data['description'].tolist()
product_titles = product_data['title'].tolist()
objections = objections_data['objection'].tolist()
responses = objections_data['response'].tolist()

# Initialize models
@st.cache_resource
def initialize_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vosk_model_path = os.getenv("VOSK_MODEL_PATH")
    vosk_model = vosk.Model(vosk_model_path)
    return model, vosk_model

model, vosk_model = initialize_models()

# Create embeddings and FAISS indices
@st.cache_resource
def create_indices():
    product_embeddings = model.encode(product_descriptions)
    objection_embeddings = model.encode(objections)

    product_index = faiss.IndexFlatL2(product_embeddings.shape[1])
    product_index.add(product_embeddings)

    objection_index = faiss.IndexFlatL2(objection_embeddings.shape[1])
    objection_index.add(objection_embeddings)

    return product_index, objection_index

product_index, objection_index = create_indices()

# Initialize audio stream
def initialize_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
    recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
    return audio, stream, recognizer

# Hugging Face API for sentiment analysis
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
API_KEY = os.getenv("API_KEY")
headers = {"Authorization": f"Bearer {API_KEY}"}

def analyze_sentiment(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        
        # Extracting the sentiment label with the highest score
        sentiments = result[0]  # Access the first element of the list
        if len(sentiments) > 0:
            # Find the sentiment with the highest score
            best_sentiment = max(sentiments, key=lambda x: x['score'])
            return best_sentiment
        else:
            return {"label": "ERROR", "score": 0.0}
    return {"label": "ERROR", "score": 0.0}


# Google Sheets API setup
@st.cache_resource
def setup_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials_path = os.getenv("GOOGLE_SHEET_CREDENTIALS_PATH")
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(creds)
    sheet = client.open("sheet").sheet1
    return sheet

sheet = setup_google_sheets()

def append_to_sheet(sentiment, transcription):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([timestamp, sentiment['label'], sentiment['score'], transcription])

def recommend_products(query):
    query_embedding = model.encode([query])
    distances, indices = product_index.search(query_embedding, 3)
    return [(product_titles[i], product_descriptions[i]) for i in indices[0]]

def handle_objection(query):
    query_embedding = model.encode([query])
    distances, indices = objection_index.search(query_embedding, 1)
    idx = indices[0][0]
    return objections[idx], responses[idx]

# Streamlit UI
st.title("Real-Time Product Recommendation & Sentiment Analysis")

if st.button("Start Listening"):
    st.info("Listening... Speak into the microphone.")
    audio, stream, recognizer = initialize_audio()

    try:
        while True:
            data = stream.read(4000)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                transcription = result.get("text", "")

                if transcription.strip():
                    st.write(f"User: {transcription}")

                    # Product Recommendations
                    recommendations = recommend_products(transcription)
                    st.subheader("Product Recommendations")
                    for title, description in recommendations:
                        st.write(f"- **{title}**: {description}")

                    # Objection Handling
                    objection, response = handle_objection(transcription)
                    st.subheader("Objection Handling")
                    st.write(f"**Objection:** {objection}")
                    st.write(f"**Response:** {response}")

                    # Sentiment Analysis
                    sentiment = analyze_sentiment(transcription)
                    st.subheader("Sentiment Analysis")
                    st.write(f"**Sentiment:** {sentiment['label']}")
                    st.write(f"**Score:** {sentiment['score']}")

                    # Save to Google Sheets
                    append_to_sheet(sentiment, transcription)
                    st.success("Data saved to Google Sheets.")

    except KeyboardInterrupt:
        st.warning("Stopped listening.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
