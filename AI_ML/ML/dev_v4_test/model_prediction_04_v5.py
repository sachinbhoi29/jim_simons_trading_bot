# only sector column is added and processed
import pandas as pd
import numpy as np
import joblib

# ===============================
# ⚙️ CONFIGURATION
# ===============================
MODEL_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4_test/models/"
DATA_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4_test/data/normalized_data_for_ml.csv"
TRADES_SAVE_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4_test/data/"

# FIXED threshold on model ensemble probability to select signals
TARGET_THRESHOLD = 0.005
THRESHOLD = 0.613              # FIXED threshold for selecting trades (model-derived)

# ===============================
# 1️⃣ LOAD DATA
# ===============================
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["Ticker"] = df["Ticker"].astype(str)

# Ensure the label column exists for later evaluation, but do NOT use it as a feature
if "future_return" not in df.columns:
    raise ValueError("DATA_PATH must include 'future_return' for evaluation, but it will NOT be used as a feature.")

# Exclude label/date/ticker from features (no leakage)
exclude = ["Date", "Ticker", "future_return"]
features = [c for c in df.columns if c not in exclude]

X = df[features]
print(f"Total samples: {len(X)}")

# ===============================
# 2️⃣ LOAD MODELS
# ===============================
print("\nLoading trained models...")
xgb_3d4 = joblib.load(MODEL_PATH + "XGBoost_model_highconf_L_5d_5P_sector.pkl")
lgb_3d4 = joblib.load(MODEL_PATH + "LightGBM_model_highconf_L_5d_5P_sector.pkl")
cat_3d4 = joblib.load(MODEL_PATH + "CatBoost_model_highconf_L_5d_5P_sector.pkl")

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
# 4️⃣ APPLY FIXED PROBABILITY THRESHOLD (NO LABEL FILTERING)
# ===============================
df["pred"] = (df["prob"] >= THRESHOLD).astype(int)
selected = df[df["pred"] == 1].copy()

print(f"\nTrades selected (prob >= {THRESHOLD}): {len(selected)}")

# ===============================
# 5️⃣ EVALUATION (LABELS USED ONLY FOR METRICS — NO FILTERING)
# ===============================
# Define "actual positive" for evaluation (no leakage into selection)
# Here we use future_return > 0 as the ground-truth success. Change if you want a different rule.
df["actual_positive"] = (df["future_return"] > TARGET_THRESHOLD).astype(int)
selected["actual_positive"] = (selected["future_return"] > TARGET_THRESHOLD).astype(int)

tp = int(((selected["actual_positive"] == 1)).sum())
fp = int(((selected["actual_positive"] == 0)).sum())
total = len(selected)

precision_final = tp / (tp + fp) if total > 0 else 0.0
recall_final = tp / df["actual_positive"].sum() if df["actual_positive"].sum() > 0 else 0.0

# Win / loss stats on selected trades (use future_return directly)
wins = selected[selected["future_return"] > TARGET_THRESHOLD]["future_return"]
losses = selected[selected["future_return"] <= TARGET_THRESHOLD]["future_return"]

avg_win = wins.mean() if len(wins) > 0 else 0.0
avg_loss = -losses.mean() if len(losses) > 0 else 0.0

P_win = len(wins) / total if total > 0 else 0.0
P_loss = len(losses) / total if total > 0 else 0.0

# Normalized expectation using raw returns (not percentage)
normalized_expectation = P_win * avg_win - P_loss * avg_loss

total_future_returns = selected["future_return"].sum()
avg_return_per_trade_pct = (total_future_returns / total * 100.0) if total > 0 else 0.0

# ===============================
# 📊 PRINT FULL SUMMARY
# ===============================
print("\n================= HIGH-CONFIDENCE TRADE SUMMARY =================")
print(f"Percent of all opportunities taken: {total / len(df) * 100:.2f}%")
print(f"Total trades selected: {total}")
print(f"True Positives (future_return>0): {tp}")
print(f"False Positives (future_return<=0): {fp}")
print(f"Precision: {precision_final:.4f}")
print(f"Recall: {recall_final:.4f}")
if fp > 0:
    print(f"TP-to-FP ratio: {tp/fp:.2f}")

print(f"\nAverage win (future_return): {avg_win:.6f}")
print(f"Average loss (abs future_return): {avg_loss:.6f}")
print(f"Normalized Expectation (raw future_return units): {normalized_expectation:.6f}")
print(f"Sum future returns: {total_future_returns:.6f}")
print(f"Avg future return per trade (percent): {avg_return_per_trade_pct:.2f}%")
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
