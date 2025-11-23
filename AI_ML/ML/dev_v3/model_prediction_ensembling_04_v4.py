import pandas as pd
import numpy as np
import joblib

# ===============================
# ⚙️ CONFIGURATION
# ===============================
MODEL_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v3/models/"
DATA_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v3/data/normalized_data_for_ml.csv"
TRADES_SAVE_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v3/data/"

TARGET_THRESHOLD = 0.005      # used ONLY for evaluation (NOT filtering)
THRESHOLD = 0.70              # FIXED threshold for selecting trades (NO optimization)

# ===============================
# 1️⃣ LOAD DATA
# ===============================
print("Loading data...")

df = pd.read_csv(DATA_PATH)
df["Ticker"] = df["Ticker"].astype(str)

# This is only for measuring prediction quality AFTER prediction
df["target_bin"] = (df["future_return"] > TARGET_THRESHOLD).astype(int)

exclude = ["Date", "Ticker", "future_return", "target_bin"]
features = [c for c in df.columns if c not in exclude]

X = df[features]

print(f"Total samples: {len(X)} | Positive rate: {df['target_bin'].mean():.4f}")

# ===============================
# 2️⃣ LOAD MODELS
# ===============================
print("\nLoading trained models...")

xgb_3d4 = joblib.load(MODEL_PATH + "XGBoost_model_highconf.pkl")
lgb_3d4 = joblib.load(MODEL_PATH + "LightGBM_model_highconf.pkl")
cat_3d4 = joblib.load(MODEL_PATH + "CatBoost_model_highconf.pkl")

# ===============================
# 3️⃣ GENERATE ENSEMBLE PROBABILITIES
# ===============================
print("\nGenerating ensemble probabilities...")

probs = np.column_stack([
    xgb_3d4.predict_proba(X)[:, 1],
    lgb_3d4.predict_proba(X)[:, 1],
    cat_3d4.predict_proba(X)[:, 1]
])

df["prob"] = probs.mean(axis=1)

print("\nProbability distribution:")
print(df["prob"].describe(percentiles=[0.9, 0.95, 0.99]))

# ===============================
# 4️⃣ APPLY FIXED THRESHOLD (NO label filtering)
# ===============================
df["pred"] = (df["prob"] >= THRESHOLD).astype(int)
selected = df[df["pred"] == 1]

print(f"\nTrades selected: {len(selected)}")

# ===============================
# 5️⃣ FULL PREDICTION STATS (NO filtering)
# ===============================
tp = (selected["target_bin"] == 1).sum()
fp = (selected["target_bin"] == 0).sum()
total = len(selected)

precision_final = tp / (tp + fp) if total > 0 else 0
recall_final = tp / df["target_bin"].sum() if df["target_bin"].sum() > 0 else 0

# --- Win / loss stats ---
wins = selected[selected["future_return"] > 0]["future_return"]
losses = selected[selected["future_return"] <= 0]["future_return"]

avg_win = wins.mean() if len(wins) > 0 else 0
avg_loss = -losses.mean() if len(losses) > 0 else 0

P_win = len(wins) / total if total > 0 else 0
P_loss = len(losses) / total if total > 0 else 0

normalized_expectation = P_win * avg_win - P_loss * avg_loss

total_future_returns = selected["future_return"].sum()

# ===============================
# 📊 PRINT FULL SUMMARY
# ===============================
print("\n================= HIGH-CONFIDENCE TRADE SUMMARY =================")
print(f"Percent of all opportunities taken: {total / len(df) * 100:.2f}%")
print(f"Total trades selected: {total}")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"Precision: {precision_final:.4f}")
print(f"Recall: {recall_final:.4f}")
if fp > 0:
    print(f"TP-to-FP ratio: {tp/fp:.2f}")

print(f"\nAverage win: {avg_win:.4f}")
print(f"Average loss: {avg_loss:.4f}")
print(f"Normalized Expectation: {normalized_expectation:.4f}")
print(f"Sum future returns: {total_future_returns:.4f}")
print(f"Avg return per trade: {total_future_returns/total*100:.2f}%")
print("==================================================================")

# ===============================
# 6️⃣ SAVE TRADES
# ===============================
selected_sorted = selected.sort_values(by="prob", ascending=False)
output_path = TRADES_SAVE_PATH + "Ensemble_highconf_trades_prediction_only_stats.csv"
selected_sorted.to_csv(output_path, index=False)

print(f"\nSaved selected trades to: {output_path}")
print("\nTop 10 trades:")
print(selected_sorted[["Date", "Ticker", "prob", "future_return"]].head(10))
