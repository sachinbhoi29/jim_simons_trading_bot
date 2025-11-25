import pandas as pd
import numpy as np
import joblib

# ===============================
# ⚙️ CONFIGURATION
# ===============================
MODEL_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4_test/models/"
DATA_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4_test/data/normalized_data_for_ml.csv"
TRADES_SAVE_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4_test/data/"

TARGET_THRESHOLD = 0.005
THRESHOLD = 0.54       # fixed probability cutoff

# ===============================
# 1️⃣ LOAD DATA
# ===============================
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["Ticker"] = df["Ticker"].astype(str)

if "future_return" not in df.columns:
    raise ValueError("DATA_PATH must include 'future_return' for evaluation.")

exclude = ["Date", "Ticker", "future_return"]
features = [c for c in df.columns if c not in exclude]

X = df[features]
print(f"Total samples: {len(X)}")

# ===============================
# 2️⃣ LOAD 6 MODELS (3D4P + 5D5P)
# ===============================
print("\nLoading trained models...")

model_files = {
    # 3D 4P MODELS
    "xgb_3d4": "XGBoost_model_highconf_gridsearch_optimized_3d_4p_v1_long.pkl",
    "lgb_3d4": "LightGBM_model_highconf_gridsearch_optimized_3d_4p_v1_long.pkl",
    "cat_3d4": "CatBoost_model_highconf_gridsearch_optimized_3d_4p_v1_long.pkl",

    # 5D 5P MODELS (YOUR NEW MODELS)
    "xgb_5d5": "XGBoost_model_highconf_gridsearch_optimized_5d_5p_v1_long.pkl",
    "lgb_5d5": "LightGBM_model_highconf_gridsearch_optimized_5d_5p_v1_long.pkl",
    "cat_5d5": "CatBoost_model_highconf_gridsearch_optimized_5d_5p_v1_long.pkl",
}

models = {}

for name, file in model_files.items():
    print(f"Loading {name} ...")
    models[name] = joblib.load(MODEL_PATH + file)

print(f"\nTotal models loaded: {len(models)}")

# ===============================
# 3️⃣ GENERATE ENSEMBLE PROBABILITIES (6-model mean)
# ===============================
print("\nGenerating ensemble probabilities...")

model_prob_list = []
for name, model in models.items():
    prob = model.predict_proba(X)[:, 1]
    model_prob_list.append(prob)
    df[f"prob_{name}"] = prob  # optional: save individual contributions

probs = np.column_stack(model_prob_list)

weights = {
    "xgb_3d4": 1.0,
    "lgb_3d4": 1.0,
    "cat_3d4": 1.0,
    "xgb_5d5": 1.0,
    "lgb_5d5": 1.0,
    "cat_5d5": 1.0,}

# Create a weighted sum of model probabilities
weighted_probs = np.zeros(len(df))

total_weight = 0

for i, (name, model) in enumerate(models.items()):
    w = weights[name]
    weighted_probs += model_prob_list[i] * w
    total_weight += w

df["prob"] = weighted_probs / total_weight

print("\nEnsemble probability distribution:")
print(df["prob"].describe(percentiles=[0.9, 0.95, 0.99]))

# ===============================
# 4️⃣ APPLY PROBABILITY THRESHOLD
# ===============================
df["pred"] = (df["prob"] >= THRESHOLD).astype(int)
selected = df[df["pred"] == 1].copy()

print(f"\nTrades selected (prob >= {THRESHOLD}): {len(selected)}")

# ===============================
# 5️⃣ EVALUATION
# ===============================
df["actual_positive"] = (df["future_return"] > TARGET_THRESHOLD).astype(int)
selected["actual_positive"] = (selected["future_return"] > TARGET_THRESHOLD).astype(int)

tp = int((selected["actual_positive"] == 1).sum())
fp = int((selected["actual_positive"] == 0).sum())
total = len(selected)

precision_final = tp / (tp + fp) if total > 0 else 0.0
recall_final = tp / df["actual_positive"].sum() if df["actual_positive"].sum() > 0 else 0.0

wins = selected[selected["future_return"] > TARGET_THRESHOLD]["future_return"]
losses = selected[selected["future_return"] <= TARGET_THRESHOLD]["future_return"]

avg_win = wins.mean() if len(wins) > 0 else 0.0
avg_loss = -losses.mean() if len(losses) > 0 else 0.0

P_win = len(wins) / total if total > 0 else 0.0
P_loss = len(losses) / total if total > 0 else 0.0

normalized_expectation = P_win * avg_win - P_loss * avg_loss

total_future_returns = selected["future_return"].sum()
avg_return_per_trade_pct = (total_future_returns / total * 100.0) if total > 0 else 0.0

# ===============================
# 📊 SUMMARY
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

print(f"\nAverage win: {avg_win:.6f}")
print(f"Average loss: {avg_loss:.6f}")
print(f"Normalized Expectation: {normalized_expectation:.6f}")
print(f"Sum future returns: {total_future_returns:.6f}")
print(f"Avg future return per trade: {avg_return_per_trade_pct:.2f}%")
print("==================================================================")

# ===============================
# 6️⃣ SAVE TRADES
# ===============================
selected_sorted = selected.sort_values(by="prob", ascending=False)
output_path = TRADES_SAVE_PATH + "Ensemble_highconf_trades_prediction_only_stats_6model.csv"
selected_sorted.to_csv(output_path, index=False)

print(f"\nSaved selected trades to: {output_path}")
print("\nTop 10 trades:")
print(selected_sorted[["Date", "Ticker", "prob", "future_return"]].head(10))
