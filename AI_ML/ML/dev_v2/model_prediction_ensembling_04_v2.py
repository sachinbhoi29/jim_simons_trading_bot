# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.metrics import precision_score

# # ===============================
# # ⚙️ CONFIGURATION (ALL PARAMETERS HERE)
# # ===============================
# MODEL_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/models/"
# DATA_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/data/normalized_data_for_ml.csv"
# TRADES_SAVE_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/data/"

# TARGET_THRESHOLD = 0.001      # Minimum future return to count as positive
# PRECISION_FLOOR = 0.50        # Minimum acceptable precision for threshold selection
# MIN_TRADES = 500              # Minimum trades at each threshold to consider
# TOP_LIMIT = None               # Max number of trades to select (None = no limit)
# THRESHOLD_SEARCH_STEPS = 50   # Number of candidate thresholds to scan between 0.5-0.99
# THRESHOLD = 0.50              # Threshold for high-confidence trades               
# # ===============================
# # 1️⃣ LOAD DATA
# # ===============================
# print("Loading data...")
# df = pd.read_csv(DATA_PATH)
# df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
# df["Ticker"] = df["Ticker"].astype(str)
# df["target_bin"] = (df["future_return"] > TARGET_THRESHOLD).astype(int)

# exclude = ["Date", "Ticker", "future_return", "target_bin"]
# features = [c for c in df.columns if c not in exclude]

# X = df[features]
# y = df["target_bin"]

# print(f"Total samples: {len(X)} | Positive rate: {y.mean():.4f}")

# # ===============================
# # 2️⃣ LOAD MODELS
# # ===============================
# print("\nLoading trained models...")
# # best_xgb = joblib.load(MODEL_PATH + "XGBoost_model_highconf_gridsearch_optimized_3d_1p_bull.pkl")
# # best_lgb = joblib.load(MODEL_PATH + "LightGBM_model_highconf_gridsearch_optimized_3d_1p_bull.pkl")
# # best_cat = joblib.load(MODEL_PATH + "CatBoost_model_highconf_gridsearch_optimized_3d_1p_bull.pkl")
# best_xgb = joblib.load(MODEL_PATH + "XGBoost_model_highconf.pkl")
# best_lgb = joblib.load(MODEL_PATH + "LightGBM_model_highconf.pkl")
# best_cat = joblib.load(MODEL_PATH + "CatBoost_model_highconf.pkl")


# # ===============================
# # 3️⃣ GENERATE ENSEMBLE PROBABILITIES
# # ===============================
# print("\nGenerating ensemble probabilities...")
# xgb_prob = best_xgb.predict_proba(X)[:, 1]
# lgb_prob = best_lgb.predict_proba(X)[:, 1]
# cat_prob = best_cat.predict_proba(X)[:, 1]
# df["prob"] = (xgb_prob + lgb_prob + cat_prob) / 3

# print("\nTop probability stats:\n", df["prob"].describe(percentiles=[0.9, 0.95, 0.99]))

# # ===============================
# # 4️⃣ THRESHOLD SELECTION (HIGH-PRECISION)
# # ===============================
# print("\nOptimizing threshold for high-confidence trades...")
# precisions = []
# thresholds = np.linspace(0.99, THRESHOLD, THRESHOLD_SEARCH_STEPS)

# for t in thresholds:
#     preds = (df["prob"] >= t).astype(int)
#     n_trades = preds.sum()
#     if n_trades < MIN_TRADES:
#         continue
#     prec = precision_score(df["target_bin"], preds)
#     rec = (preds & df["target_bin"]).sum() / df["target_bin"].sum()
#     precisions.append((t, prec, rec, n_trades))

# if precisions:
#     precisions_df = pd.DataFrame(precisions, columns=["threshold", "precision", "recall", "n_trades"])
#     valid = precisions_df[precisions_df["precision"] >= PRECISION_FLOOR]
#     if not valid.empty:
#         best_row = valid.sort_values(by="n_trades", ascending=False).iloc[0]
#         best_threshold = best_row["threshold"]
#     else:
#         best_threshold = precisions_df.sort_values(by="precision", ascending=False).iloc[0]["threshold"]
# else:
#     print("\nNo threshold meets minimum trade requirement. Using default 0.9")
#     best_threshold = 0.9

# print("\n===== Threshold optimization =====")
# if precisions:
#     print(precisions_df.head(10))
# print(f"\nSelected threshold for high-confidence trades: {best_threshold:.3f}")

# # ===============================
# # 5️⃣ FINAL SELECTION
# # ===============================
# df["pred"] = (df["prob"] >= best_threshold).astype(int)
# selected = df[df["pred"] == 1]

# # Limit to top trades if TOP_LIMIT is set
# if TOP_LIMIT is not None:
#     selected = selected.head(TOP_LIMIT)

# tp = ((selected["target_bin"] == 1).sum())
# fp = ((selected["target_bin"] == 0).sum())
# total = len(selected)
# precision_final = tp / (tp + fp) if (tp + fp) > 0 else 0
# recall_final = tp / df["target_bin"].sum() if df["target_bin"].sum() > 0 else 0

# print("\n================= HIGH-CONFIDENCE TRADE SUMMARY =================")
# percent_taken = total / len(df) * 100
# print(f"Percent of all opportunities taken: {percent_taken:.2f}%")
# print(f"Total trades selected: {total}")
# print(f"True Positives (real winners): {tp}")
# print(f"False Positives (fake signals): {fp}")
# print(f"Precision (TP / TP+FP): {precision_final:.4f}")
# print(f"Recall: {recall_final:.4f}")
# if fp > 0:
#     print(f"TP-to-FP ratio: {tp/fp:.2f}")

# print("==================================================================")

# # ===============================
# # 6️⃣ SAVE RESULTS
# # ===============================
# selected_sorted = selected.sort_values(by="prob", ascending=False)
# output_path = TRADES_SAVE_PATH + "Ensemble_highconf_trades.csv"
# selected_sorted.to_csv(output_path, index=False)

# print(f"\nSaved {len(selected_sorted)} ultra-high-confidence trades to: {output_path}")
# print("\nTop 10 high-confidence trades:")
# print(selected_sorted[["Date", "Ticker", "prob", "future_return"]].head(10))
