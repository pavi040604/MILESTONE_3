import os
import sys
import vosk
import pandas as pd
import pyaudio
import requests
import gspread
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import RagTokenizer, RagTokenForGeneration
from transformers import DPRQuestionEncoderTokenizerFast, BartTokenizerFast
import faiss
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# File paths from environment variables
PRODUCT_DATA_FILE = os.getenv("PRODUCT_DATA_FILE")
OBJECTIONS_DATA_FILE = os.getenv("OBJECTIONS_DATA_FILE")
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH")
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
API_URL = os.getenv("API_URL", "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english")

# Load product and objection data
print("Loading product data...")
product_data = pd.read_csv(PRODUCT_DATA_FILE)
product_descriptions = product_data['description'].tolist()
product_titles = product_data['title'].tolist()

print("Loading objections data...")
objections_data = pd.read_csv(OBJECTIONS_DATA_FILE)
objections = objections_data['objection'].tolist()
responses = objections_data['response'].tolist()

# Load Sentence Transformer model
print("Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for product descriptions and objections
print("Generating embeddings...")
product_embeddings = model.encode(product_descriptions)
objection_embeddings = model.encode(objections)

# Create FAISS indices for products and objections
print("Creating FAISS indices...")
product_index = faiss.IndexFlatL2(product_embeddings.shape[1])
product_index.add(product_embeddings)

objection_index = faiss.IndexFlatL2(objection_embeddings.shape[1])
objection_index.add(objection_embeddings)

# Initialize Vosk Model
print("Loading Vosk model...")
vosk_model = vosk.Model(VOSK_MODEL_PATH)
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)

# Initialize pyaudio
print("Initializing audio stream...")
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)

# Hugging Face API headers
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

def analyze_sentiment(text):
    """Analyze sentiment using Hugging Face API."""
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        elif isinstance(result, dict):
            return result
        else:
            print("Unexpected response format:", result)
            return {"label": "ERROR", "score": 0.0}
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return {"label": "ERROR", "score": 0.0}

# Google Sheets API Setup
print("Setting up Google Sheets API...")
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDENTIALS_FILE, scope)
client = gspread.authorize(creds)
sheet = client.open("sheet").sheet1

def append_to_sheet(sentiment, transcription):
    """Append the sentiment and transcription to Google Sheets."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sentiment_label = sentiment.get('label', 'Unknown') if isinstance(sentiment, dict) else 'Unknown'
    sentiment_score = sentiment.get('score', 0.0) if isinstance(sentiment, dict) else 0.0
    sheet.append_row([timestamp, sentiment_label, sentiment_score, transcription])

# Remaining logic unchanged...
