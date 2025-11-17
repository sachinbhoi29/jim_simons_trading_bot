# import pandas as pd
# import numpy as np


# def preprocess_derived_features_v2(df: pd.DataFrame, smooth=True) -> pd.DataFrame:
#     """
#     v2 (no scaling version): Stable preprocessing pipeline
#     - Uses only relative/log/diff features
#     - No StandardScaler needed
#     - Deterministic output
#     """

#     # ====================================================
#     # 1️⃣ Relative Price-Based Normalization
#     # ====================================================
#     price_cols = [
#         "EMA_20", "EMA_50", "BB_mid_20", "BB_upper_20", "BB_lower_20",
#         "EMA_fast", "EMA_slow", "VWAP",
#         "NIFTY_EMA_20", "NIFTY_EMA_50",
#         "BANKNIFTY_EMA_20", "BANKNIFTY_EMA_50"
#     ]

#     for col in price_cols:
#         if col in df.columns:
#             if "BANKNIFTY" in col:
#                 base = "BANKNIFTY_Close"
#             elif "NIFTY" in col:
#                 base = "NIFTY_Close"
#             else:
#                 base = "Close"
#             df[col + "_rel"] = (df[col] / df[base]) - 1

#     # ====================================================
#     # 2️⃣ Log Transform Volumes
#     # ====================================================
#     volume_cols = [
#         "Volume", "Cum_Vol", "Cum_TPV",
#         "NIFTY_Volume", "NIFTY_Cum_Vol",
#         "BANKNIFTY_Volume", "BANKNIFTY_Cum_Vol"
#     ]

#     for col in volume_cols:
#         if col in df.columns:
#             df[col + "_log"] = np.log1p(df[col])

#     # ====================================================
#     # 3️⃣ Encode Regimes
#     # ====================================================
#     mapping = {"Bearish": -1, "Neutral": 0, "Bullish": 1, "High Volatility": 2}

#     for col in ["NIFTY_Enhanced_Regime_for_RD", "BANKNIFTY_Enhanced_Regime_for_RD"]:
#         if col in df.columns:
#             df[col] = df[col].map(mapping).fillna(0).astype(int)

#     # ====================================================
#     # 4️⃣ Optional Smoothing
#     # ====================================================
#     if smooth:
#         smooth_cols = [
#             "MACD_diff", "EMA_diff", "VWAP_diff",
#             "RSI_diff_NIFTY", "RSI_diff_BANKNIFTY",
#             "ATR_vs_NIFTY", "ATR_vs_BANKNIFTY"
#         ]

#         for col in smooth_cols:
#             if col in df.columns:
#                 df[col] = df.groupby("Ticker")[col].transform(
#                     lambda x: x.rolling(3, min_periods=1).mean()
#                 )

#     # ====================================================
#     # 5️⃣ Trim invalid rows
#     # ====================================================
#     critical = [
#         c for c in df.columns
#         if ("_rel" in c) or (c in ["ATR_14", "Range_pct"])
#     ]

#     cleaned = []
#     for ticker, group in df.groupby("Ticker", group_keys=False):
#         mask = group[critical].notna().all(axis=1)
#         if mask.any():
#             first = mask.idxmax()
#             group = group.loc[first:]
#         cleaned.append(group)

#     df = pd.concat(cleaned, ignore_index=True)

#     # trim invalid futures
#     cleaned2 = []
#     for ticker, group in df.groupby("Ticker", group_keys=False):
#         mask = group["future_return"].notna()
#         if mask.any():
#             last = mask[::-1].idxmax()
#             group = group.loc[:last]
#         cleaned2.append(group)

#     df = pd.concat(cleaned2, ignore_index=True)
#     df = df[df["future_return"].notna()]
#     subset_cols = [
#     "future_return", "EMA_diff", "EMA_20_rel", "EMA_50_rel", "VWAP_rel",
#     "BB_mid_20_rel", "BB_upper_20_rel", "BB_lower_20_rel",
#     "NIFTY_EMA_20_rel", "NIFTY_EMA_50_rel",
#     "BANKNIFTY_EMA_20_rel", "BANKNIFTY_EMA_50_rel",
#     "MACD_diff", "RSI_diff_NIFTY", "RSI_diff_BANKNIFTY",
#     "VWAP_diff", "VWAP_NIFTY_diff", "VWAP_BANKNIFTY_diff", #they differ, comment them
#     "BB_Width", "BB_Pos", "NIFTY_BB_Width", "BANKNIFTY_BB_Width",
#     "Close_vs_EMA20", "Close_vs_EMA50", "BB_Mid_Dev",
#     "RSI_vs_MACD", "RSI_vs_NIFTY", "RSI_vs_BANKNIFTY",
#     "Range_over_ATR", "Range_over_Close",
#     "VWAP_minus_Close", "VWAP_minus_EMA20",
#     "ATR_vs_NIFTY", "ATR_vs_BANKNIFTY", "log_ATR_vs_NIFTY",
#     "Range_pct", "RSI_14", "MACD", "ATR_14",
#     "Volume_log", "Cum_Vol_log", "Cum_TPV_log",
#     "NIFTY_Volume_log", "NIFTY_Cum_Vol_log",
#     "BANKNIFTY_Volume_log", "BANKNIFTY_Cum_Vol_log",
#     "NIFTY_Enhanced_Regime_for_RD",
#     "BANKNIFTY_Enhanced_Regime_for_RD"]
#     df = df.dropna(subset=subset_cols)




#     # ====================================================
#     # 6️⃣ Select ONLY required ML Columns
#     # ====================================================
#     ml_cols = [
#         "Date", "Ticker", "future_return",

#         # Relatives
#         "EMA_diff", "EMA_20_rel", "EMA_50_rel", "VWAP_rel",
#         "BB_mid_20_rel", "BB_upper_20_rel", "BB_lower_20_rel",
#         "NIFTY_EMA_20_rel", "NIFTY_EMA_50_rel",
#         "BANKNIFTY_EMA_20_rel", "BANKNIFTY_EMA_50_rel",

#         # Derived indicators
#         "MACD_diff", "RSI_diff_NIFTY", "RSI_diff_BANKNIFTY",
#         "VWAP_diff", "VWAP_NIFTY_diff", "VWAP_BANKNIFTY_diff",
#         "BB_Width", "BB_Pos", "NIFTY_BB_Width", "BANKNIFTY_BB_Width",
#         "Close_vs_EMA20", "Close_vs_EMA50", "BB_Mid_Dev",
#         "RSI_vs_MACD", "RSI_vs_NIFTY", "RSI_vs_BANKNIFTY",
#         "Range_over_ATR", "Range_over_Close", "VWAP_minus_Close",
#         "VWAP_minus_EMA20", "ATR_vs_NIFTY", "ATR_vs_BANKNIFTY",
#         "log_ATR_vs_NIFTY",

#         # Raw core indicators
#         "Range_pct", "RSI_14", "MACD", "ATR_14",

#         # Log Volumes
#         "Volume_log", "Cum_Vol_log", "Cum_TPV_log",
#         "NIFTY_Volume_log", "NIFTY_Cum_Vol_log",
#         "BANKNIFTY_Volume_log", "BANKNIFTY_Cum_Vol_log",

#         # Regimes
#         "NIFTY_Enhanced_Regime_for_RD",
#         "BANKNIFTY_Enhanced_Regime_for_RD"
#     ]

#     df_ml = df[[c for c in ml_cols if c in df.columns]].copy()

#     # ====================================================
#     # 7️⃣ FFILL/BFILL
#     # ====================================================
#     num = df_ml.select_dtypes(include=[np.number]).columns

#     for ticker, group in df_ml.groupby("Ticker", group_keys=False):
#         df_ml.loc[group.index, num] = (
#             group[num]
#             .fillna(method="ffill", limit=1)
#             .fillna(method="bfill", limit=1)
#         )

#     print(f"Preprocessing v2 (no scaling) | Features: {df_ml.shape[1]} | Rows: {df_ml.shape[0]}")
#     return df_ml


# # ========================================
# # Script Entry Point (NO SCALING)
# # ========================================
# if __name__ == "__main__":

#     input_path = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v3/data/features_ready.csv"
#     output_path = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v3/data/normalized_data_for_ml.csv"

#     print("Loading raw derived features...")
#     df_raw = pd.read_csv(input_path)

#     print("Running preprocessing v2 (no scaling)...")
#     df_ml = preprocess_derived_features_v2(df_raw)

#     print(f"Saving ML dataset to: {output_path}")
#     df_ml.to_csv(output_path, index=False)

#     print("Done.")
