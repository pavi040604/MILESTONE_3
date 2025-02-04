# SAKES CALL ASSITANT

## Overview
This project provides a **Call Sentiment Analysis  , product Recommendation, Objection Handling and  Dashboard** using **Streamlit** 
for interactive visualization. It leverages **machine learning models** for sentiment analysis, **product recommendations**, and **objection handling** 
while integrating **speech recognition** and **text processing** models.

## Features
- **Sentiment Analysis:** Analyzes customer sentiment using **Hugging Face DistilBERT** API.
- **Predictive Sentiment Trends:** Uses **Linear Regression** to predict future sentiment trends.
- **Word Cloud Visualization:** Displays frequent topics discussed in calls.
- **Product Recommendation:** Identifies relevant products based on call transcripts.
- **Objection Handling:** Provides responses to customer objections.
- **Speech-to-Text Processing:** Uses **Vosk** for real-time transcription.

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed. Install dependencies using:
```sh
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file and define the following environment variables:
```
API_KEY=your_huggingface_api_key
MODEL_PATH=path_to_vosk_model
PROD_PATH=path_to_product_data.csv
OBJ_PATH=path_to_objections_data.csv
```

## Usage
### Running the Application
Start the **Streamlit dashboard** by executing:
```sh
streamlit run app.py
```

### Input Data Format
The `session_data.json` should be structured as:
```json
{
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "interactions": [
    {
      "transcription": "Customer call transcription",
      "sentiment": [{ "label": "positive", "score": 0.92 }],
      "product_recommendations": [["Product A", "Description A"]],
      "objection_handling": { "objection": "Price is too high", "response": "We offer discounts" }
    }
  ]
}
```

## Project Structure
```
├── dashboard.py  # Data analysis and visualization
├── app.py        # Streamlit dashboard logic
├── requirements.txt  # Python dependencies
├── .env          # Environment variables
```

## Technologies Used
- **Python**, **Pandas**, **NumPy**
- **Streamlit** (Interactive Dashboard)
- **Matplotlib** (Data Visualization)
- **Scikit-Learn** (Machine Learning for sentiment prediction)
- **Vosk** (Speech Recognition)
- **Hugging Face Transformers** (NLP Models)
- **FAISS** (Efficient similarity search)
- **Google Sheets API** (Data storage & retrieval)

## Future Enhancements
- **More advanced sentiment analysis models**
- **Improved objection handling with GPT-based models**
- **Real-time speech-to-text improvements**

For any queries, feel free to contact the project owner.

