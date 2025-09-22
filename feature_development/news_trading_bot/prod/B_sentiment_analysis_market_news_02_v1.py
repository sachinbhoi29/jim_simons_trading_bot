import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# ------------------------------
# Step 1: Load and Clean Data
# ------------------------------

df = pd.read_csv("feature_development/news_trading_bot/prod/news_market_combined_20250919_1515_to_20250922_0915.csv")

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    text = re.sub(r'\d{1,2}\s\w+\s\d{4}', '', text)
    text = re.sub(r'[â€™â€”â€“â€œâ€\x9dâ€¦]', "'", text)
    return text.lower().strip()

df['clean_title'] = df['title'].apply(clean_text)

# ------------------------------
# Step 2: Load Improved Model
# ------------------------------

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Sentiment mapping
label_map = {
    "LABEL_0": -1,   # Negative
    "LABEL_1": 0,    # Neutral
    "LABEL_2": 1     # Positive
}

def get_sentiment_score(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
        probs = torch.nn.functional.softmax(output.logits, dim=1).numpy()[0]
        label = np.argmax(probs)
        sentiment_score = (probs[2] - probs[0]) * 100  # Positive - Negative
        return round(sentiment_score, 2)

df['sentiment_score'] = df['clean_title'].apply(get_sentiment_score)

# ------------------------------
# Step 3: Apply Custom Weights
# ------------------------------

weights = {
    'stock market': 1.0,
    'Sensex': 1.2,
    'Nifty': 1.2,
    'Federal Reserve': 1.5,
    'US Fed': 1.5,
    'inflation': 1.3,
    'crude oil': 1.1,
    'OPEC': 1.1,
    'tariffs': 1.2,
    'war': 1.4,
    'geopolitical': 1.4,
    'IPO': 0.8,
    'budget': 1.0,
    'SEBI': 1.0,
    'NSE': 1.0,
    'global markets': 1.2,
    'RBI': 1.0
}

df['weight'] = df['query'].map(weights).fillna(1.0)
df['weighted_score'] = df['sentiment_score'] * df['weight']

# ------------------------------
# Step 4: Compute Market Sentiment Index
# ------------------------------

overall_index = df['weighted_score'].sum() / df['weight'].sum()
overall_index = round(overall_index, 2)

def categorize_index(score):
    if score <= -50:
        return "Very Bearish"
    elif -50 < score <= -10:
        return "Bearish"
    elif -10 < score < 10:
        return "Neutral"
    elif 10 <= score < 50:
        return "Bullish"
    else:
        return "Very Bullish"

overall_sentiment = categorize_index(overall_index)

print(f"\n📊 Market Sentiment Index: {overall_index} → {overall_sentiment}")

# ------------------------------
# Step 5: Save Results (Optional)
# ------------------------------

df.to_csv("feature_development/news_trading_bot/prod/news_sentiment_scored.csv", index=False)
