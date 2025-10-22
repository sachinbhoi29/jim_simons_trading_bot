import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, confusion_matrix

min_trades = 1000

# ===============================
# ⚙️ Paths
# ===============================
MODEL_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/models/"
DATA_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/data/normalized_data_for_ml.csv"
TRADES_SAVE_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/data/"
TARGET_THRESHOLD = 0.002
# ===============================
# 1️⃣ Load test data
# ===============================
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Ticker'] = df['Ticker'].astype(str)

df['target_bin'] = (df['future_return'] > TARGET_THRESHOLD).astype(int)

# Assuming target is already defined
exclude = ['Date','Ticker','future_return','target_bin']
features = [c for c in df.columns if c not in exclude]

X = df[features]
y = df['target_bin']

# Split by ticker to match test set used earlier
from sklearn.model_selection import train_test_split
tickers = df['Ticker'].unique()
_, test_tickers = train_test_split(tickers, test_size=0.2, random_state=42)
test_mask = df['Ticker'].isin(test_tickers)

X_test = X[test_mask]
y_test = y[test_mask]

# ===============================
# 2️⃣ Load trained models
# ===============================
best_xgb = joblib.load(MODEL_PATH + "XGBoost_model_highconf_gridsearch_optimized.pkl")
best_lgb = joblib.load(MODEL_PATH + "LightGBM_model_highconf_gridsearch_optimized.pkl")
best_cat = joblib.load(MODEL_PATH + "CatBoost_model_highconf_gridsearch_optimized.pkl")

# ===============================
# 3️⃣ Ensemble predictions (soft voting)
# ===============================
xgb_prob = best_xgb.predict_proba(X_test)[:,1]
lgb_prob = best_lgb.predict_proba(X_test)[:,1]
cat_prob = best_cat.predict_proba(X_test)[:,1]

# Simple average
ensemble_prob = (xgb_prob + lgb_prob + cat_prob) / 3

# ===============================
# 4️⃣ Automatic threshold selection
# ===============================
def select_threshold(y_true, y_prob, min_trades=min_trades, precision_floor=0.8):
    thresholds = np.linspace(0.7, 0.99, 50)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        num_trades = y_pred.sum()
        if num_trades >= min_trades:
            prec = precision_score(y_true, y_pred)
            if prec >= precision_floor:
                return t
    return 0.5

best_threshold = select_threshold(y_test, ensemble_prob)
print(f"Ensemble high-confidence threshold: {best_threshold:.3f}")

# ===============================
# 5️⃣ Classify high-confidence trades
# ===============================
y_pred_highconf = (ensemble_prob >= best_threshold).astype(int)

precision = precision_score(y_test, y_pred_highconf)
cm = confusion_matrix(y_test, y_pred_highconf)
print("Ensemble High-Confidence Precision:", precision)
print("Confusion Matrix (TP/FP focused):\n", cm)

# ===============================
# 6️⃣ Save high-confidence trades
# ===============================
X_test_df = X_test.copy()
X_test_df['TopProb'] = ensemble_prob
X_test_df['Ticker'] = df.loc[test_mask, 'Ticker'].values
X_test_df['Date'] = df.loc[test_mask, 'Date'].values

high_conf_trades = X_test_df[X_test_df['TopProb'] >= best_threshold]
high_conf_trades_sorted = high_conf_trades.sort_values(by='TopProb', ascending=False)

# Save CSV
high_conf_trades_sorted.to_csv(TRADES_SAVE_PATH + "Ensemble_highconf_trades.csv", index=False)

print("Top 10 high-confidence trades:")
print(high_conf_trades_sorted[['Date','Ticker','TopProb']].head(10))
