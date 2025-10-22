import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# ===============================
# ⚙️ CONFIGURATION
# ===============================
MODEL_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/models/"
DATA_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/data/normalized_data_for_ml.csv"
TRADES_SAVE_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/data/"

TARGET_THRESHOLD = 0.002        # target for positive future return
MIN_TRADES = 500                # minimum trades for consideration
PRECISION_FLOOR = 0.75          # minimum acceptable precision
TOP_LIMIT = 2000                # max number of trades to consider

# ===============================
# 1️⃣ LOAD DATA
# ===============================
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Ticker"] = df["Ticker"].astype(str)
df["target_bin"] = (df["future_return"] > TARGET_THRESHOLD).astype(int)

exclude = ["Date", "Ticker", "future_return", "target_bin"]
features = [c for c in df.columns if c not in exclude]

X = df[features]
y = df["target_bin"]

# Split by ticker (same as training)
from sklearn.model_selection import train_test_split
tickers = df["Ticker"].unique()
_, test_tickers = train_test_split(tickers, test_size=0.2, random_state=42)
test_mask = df["Ticker"].isin(test_tickers)

X_test = X[test_mask].reset_index(drop=True)
y_test = y[test_mask].reset_index(drop=True)

print(f"Test samples: {len(X_test)} | Positive rate: {y_test.mean():.4f}")

# ===============================
# 2️⃣ LOAD MODELS
# ===============================
print("\nLoading trained models...")
best_xgb = joblib.load(MODEL_PATH + "XGBoost_model_highconf.pkl")
best_lgb = joblib.load(MODEL_PATH + "LightGBM_model_highconf.pkl")
best_cat = joblib.load(MODEL_PATH + "CatBoost_model_highconf.pkl")

# ===============================
# 3️⃣ ENSEMBLE PROBABILITIES
# ===============================
print("\nGenerating ensemble probabilities...")
xgb_prob = best_xgb.predict_proba(X_test)[:, 1]
lgb_prob = best_lgb.predict_proba(X_test)[:, 1]
cat_prob = best_cat.predict_proba(X_test)[:, 1]

ensemble_prob = (xgb_prob + lgb_prob + cat_prob) / 3

# Attach info for analysis
test_df = pd.DataFrame({
    "prob": ensemble_prob,
    "true": y_test,
    "Ticker": df.loc[test_mask, "Ticker"].values,
    "Date": df.loc[test_mask, "Date"].values,
    "future_return": df.loc[test_mask, "future_return"].values
})

test_df = test_df.sort_values(by="prob", ascending=False).reset_index(drop=True)
print("\nTop probability stats:\n", test_df["prob"].describe(percentiles=[0.9, 0.95, 0.99]))

# ===============================
# 4️⃣ THRESHOLD SELECTION (High Precision)
# ===============================
precisions = []
thresholds = np.linspace(0.99, 0.5, 50)

for t in thresholds:
    preds = (test_df["prob"] >= t).astype(int)
    n_trades = preds.sum()
    if n_trades < MIN_TRADES:
        continue
    prec = precision_score(test_df["true"], preds)
    rec = recall_score(test_df["true"], preds)
    precisions.append((t, prec, rec, n_trades))

if not precisions:
    print("\n⚠️ No threshold meets minimum trade requirement. Using default 0.9")
    best_threshold = 0.9
else:
    precisions_df = pd.DataFrame(precisions, columns=["threshold", "precision", "recall", "n_trades"])
    # pick threshold with precision above floor and max number of trades
    valid = precisions_df[precisions_df["precision"] >= PRECISION_FLOOR]
    if not valid.empty:
        best_row = valid.sort_values(by="n_trades", ascending=False).iloc[0]
        best_threshold = best_row["threshold"]
    else:
        best_threshold = precisions_df.sort_values(by="precision", ascending=False).iloc[0]["threshold"]

print("\n===== Threshold optimization =====")
print(precisions_df.head(10))
print(f"\nSelected threshold for high-confidence trades: {best_threshold:.3f}")

# ===============================
# 5️⃣ FINAL SELECTION
# ===============================
test_df["pred"] = (test_df["prob"] >= best_threshold).astype(int)
selected = test_df[test_df["pred"] == 1]

# Ensure we don't take too many trades
if len(selected) > TOP_LIMIT:
    selected = selected.head(TOP_LIMIT)

tp = ((selected["true"] == 1).sum())
fp = ((selected["true"] == 0).sum())
total = len(selected)
precision_final = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_final = tp / test_df["true"].sum()

print("\n================= HIGH-CONFIDENCE TRADE SUMMARY =================")
print(f"Total trades selected: {total}")
print(f"True Positives (real winners): {tp}")
print(f"False Positives (fake signals): {fp}")
print(f"Precision (TP / TP+FP): {precision_final:.4f}")
print(f"Recall: {recall_final:.4f}")
if fp > 0:
    print(f"TP-to-FP ratio: {tp/fp:.2f}")
print("==================================================================")

# ===============================
# 6️⃣ SAVE RESULTS
# ===============================
selected_sorted = selected.sort_values(by="prob", ascending=False)
output_path = TRADES_SAVE_PATH + "Ensemble_highconf_trades.csv"
selected_sorted.to_csv(output_path, index=False)

print(f"\nSaved {len(selected_sorted)} ultra-high-confidence trades to: {output_path}")
print("\nTop 10 high-confidence trades:")
print(selected_sorted[["Date", "Ticker", "prob", "future_return"]].head(10))
