import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# ------------------------------
# Step 1: Load and Clean Data
# ------------------------------

df = pd.read_csv("news_market_combined_20250820_1515_to_20250821_0915.csv")
df = df.dropna(subset=["title", "stock"])

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    text = re.sub(r'\d{1,2}\s\w+\s\d{4}', '', text)
    text = re.sub(r'[â€™â€”â€“â€œâ€\x9dâ€¦]', "'", text)
    return text.lower().strip()

df['clean_title'] = df['title'].apply(clean_text)

# ------------------------------
# Step 2: Load Sentiment Model
# ------------------------------

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def get_sentiment_score(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
        probs = torch.nn.functional.softmax(output.logits, dim=1).numpy()[0]
        sentiment_score = (probs[2] - probs[0]) * 100  # Positive - Negative
        return round(sentiment_score, 2)

print("🔍 Running sentiment analysis...")
df['sentiment_score'] = df['clean_title'].apply(get_sentiment_score)

# ------------------------------
# 🔄 NEW: Categories + Custom Weights
# ------------------------------

category_keywords = {
    "earnings": [
        "quarterly results", "q1 results", "q2 results", "q3 results", "q4 results", "quarterly ",
        "q1 ", "q2 ", "q3 ", "q4 ", "q1:", "q2:", "q3:", "q4:",
        "quarterly earnings", "net profit", "revenue", "ebitda", "eps", "financial results",
        "topline", "bottomline", "q1fy", "q2fy", "q3fy", "q4fy", "fy2025", "fy25"
    ],
    "ipo": [
        "ipo", "initial public offering", "listing", "pre-ipo", "price band", "subscription status"
    ],
    "regulation": [
        "rbi", "sebi", "regulation", "ban", "penalty", "compliance", "circular"
    ],
    "macroeconomy": [
        "inflation", "gdp", "interest rate", "repo rate", "global economy", "macroeconomic"
    ],
    "mna": [
        "acquisition", "merger", "stake buy", "stake sale", "takeover", "deal"
    ]
}

category_weights = {
    "earnings": 1.5,
    "ipo": 1.3,
    "regulation": 1.2,
    "macroeconomy": 1.1,
    "mna": 1.4,
    "general": 1.0
}

def classify_category(text):
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in text:
                return category
    return "general"

def assign_weight(category):
    return category_weights.get(category, 1.0)

df['category'] = df['clean_title'].apply(classify_category)
df['weight'] = df['category'].apply(assign_weight)
df['weighted_score'] = df['sentiment_score'] * df['weight']

# ------------------------------
# Step 4: Export Per-Article Sentiment
# ------------------------------

df.to_csv("sentiment_per_news.csv", index=False)
print("Saved per-news sentiment with category/weight: sentiment_per_news.csv")

# ------------------------------
# Step 5: Grouped Sentiment per Stock
# ------------------------------

grouped_df = df.groupby("stock").agg({
    "sentiment_score": "mean",
    "weighted_score": "mean",
    "weight": "mean",
    "title": "count"
}).reset_index()

grouped_df = grouped_df.rename(columns={
    "sentiment_score": "avg_sentiment_score",
    "weighted_score": "avg_weighted_sentiment_score",
    "weight": "avg_weight",
    "title": "article_count"
})

grouped_df["avg_sentiment_score"] = grouped_df["avg_sentiment_score"].round(2)
grouped_df["avg_weighted_sentiment_score"] = grouped_df["avg_weighted_sentiment_score"].round(2)
grouped_df["avg_weight"] = grouped_df["avg_weight"].round(2)

# ------------------------------
# Step 6: Save Aggregated Results
# ------------------------------

grouped_df.to_csv("sentiment_by_stock.csv", index=False)
print("Saved grouped sentiment: sentiment_by_stock.csv")
