import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime

# ------------------------------
# Step 1: Sentiment Model
# ------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    text = re.sub(r'\d{1,2}\s\w+\s\d{4}', '', text)
    text = re.sub(r'[â€™â€”â€“â€œâ€\x9dâ€¦]', "'", text)
    return text.lower().strip()

def get_sentiment_score(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
        probs = torch.nn.functional.softmax(output.logits, dim=1).numpy()[0]
        sentiment_score = (probs[2] - probs[0]) * 100  # Positive - Negative
        return round(sentiment_score, 2)

# ------------------------------
# Step 2: Categories + Weights
# ------------------------------
category_keywords = {
    "earnings": ["quarterly results","q1 results","q2 results","q3 results","q4 results",
                 "quarterly","net profit","revenue","ebitda","eps","financial results",
                 "topline","bottomline","fy25","fy2025"],
    "ipo": ["ipo","initial public offering","listing","pre-ipo","price band","subscription status"],
    "regulation": ["rbi","sebi","regulation","ban","penalty","compliance","circular"],
    "macroeconomy": ["inflation","gdp","interest rate","repo rate","global economy","macroeconomic"],
    "mna": ["acquisition","merger","stake buy","stake sale","takeover","deal"]
}

category_weights = {
    "earnings": 1.5, "ipo": 1.3, "regulation": 1.2,
    "macroeconomy": 1.1, "mna": 1.4, "general": 1.0
}

def classify_category(text):
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in text:
                return category
    return "general"

def assign_weight(category):
    return category_weights.get(category, 1.0)

# ------------------------------
# Step 3: Process Excel Tabs
# ------------------------------
input_file = "stock_news_aug15_sep30.xlsx"
intermediate_file = "sentiment_per_news.xlsx"
aggregated_file = "sentiment_by_date.xlsx"

xls = pd.ExcelFile(input_file)
intermediate_results = {}
aggregated_results = {}

print("🔍 Running sentiment analysis for all stocks...")

for sheet in xls.sheet_names:
    print(f"\nProcessing sheet: {sheet}")
    df = pd.read_excel(xls, sheet_name=sheet)
    df = df.dropna(subset=["title", "stock"])
    
    # Clean + sentiment
    df['clean_title'] = df['title'].apply(clean_text)
    df['sentiment_score'] = df['clean_title'].apply(get_sentiment_score)
    
    # Categorize + weight
    df['category'] = df['clean_title'].apply(classify_category)
    df['weight'] = df['category'].apply(assign_weight)
    df['weighted_score'] = df['sentiment_score'] * df['weight']
    
    # Ensure date column
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['published']).dt.date
    
    # Save per-news (intermediate)
    intermediate_results[sheet] = df
    
    # Aggregate per day
    agg = df.groupby("date").agg({
        "sentiment_score": "mean",
        "weighted_score": "mean",
        "title": "count"
    }).reset_index()
    
    agg.rename(columns={
        "sentiment_score": "mean_sentiment_score",
        "weighted_score": "mean_weighted_score",
        "title": "article_count"
    }, inplace=True)
    
    agg["mean_sentiment_score"] = agg["mean_sentiment_score"].round(2)
    agg["mean_weighted_score"] = agg["mean_weighted_score"].round(2)
    
    aggregated_results[sheet] = agg

# ------------------------------
# Step 4: Save Results
# ------------------------------
# Save intermediate (row by row)
with pd.ExcelWriter(intermediate_file, engine="openpyxl") as writer:
    for sheet, df in intermediate_results.items():
        df.to_excel(writer, sheet_name=sheet, index=False)

print(f"Saved per-news sentiment (validation): {intermediate_file}")

# Save aggregated (daily mean)
with pd.ExcelWriter(aggregated_file, engine="openpyxl") as writer:
    for sheet, df in aggregated_results.items():
        df.to_excel(writer, sheet_name=sheet, index=False)

print(f"Saved aggregated daily sentiment: {aggregated_file}")
