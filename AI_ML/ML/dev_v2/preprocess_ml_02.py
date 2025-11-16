import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_derived_features(df: pd.DataFrame, smooth=True, scale=True) -> pd.DataFrame:
    """
    Refined preprocessing for derived feature dataset:
    Per-ticker normalization
    Proper z-score computation
    Consistent scaling across features
    Optional smoothing of noisy ratios
    """

    # ====================================================
    # 1️⃣ Relative Price-Based Normalization
    # ====================================================
    price_cols = [
        "EMA_20", "EMA_50", "BB_mid_20", "BB_upper_20", "BB_lower_20",
        "EMA_fast", "EMA_slow", "VWAP",
        "NIFTY_EMA_20", "NIFTY_EMA_50",
        "BANKNIFTY_EMA_20", "BANKNIFTY_EMA_50"
    ]

    for col in price_cols:
        if col in df.columns:
            if "BANKNIFTY" in col:
                base_col = "BANKNIFTY_Close"
            elif "NIFTY" in col:
                base_col = "NIFTY_Close"
            else:
                base_col = "Close"
            df[col + "_rel"] = (df[col] / df[base_col]) - 1

    # ====================================================
    # 2️⃣ Log Transform Volume Columns
    # ====================================================
    volume_cols = [
        "Volume", "Cum_Vol", "Cum_TPV",
        "NIFTY_Volume", "NIFTY_Cum_Vol",
        "BANKNIFTY_Volume", "BANKNIFTY_Cum_Vol"
    ]
    for col in volume_cols:
        if col in df.columns:
            df[col + "_log"] = np.log1p(df[col])

    # ====================================================
    # 3️⃣ Per-Ticker Rolling Z-score Normalization
    # ====================================================
    # need to comment this as we do not have future values while predicting
    # zscore_cols = [
    #     "RSI_14", "MACD", "MACD_signal", "%K", "%D",
    #     "NIFTY_RSI_14", "NIFTY_MACD", "NIFTY_MACD_signal",
    #     "BANKNIFTY_RSI_14", "BANKNIFTY_MACD", "BANKNIFTY_MACD_signal"
    # ]
    # for col in zscore_cols:
    #     if col in df.columns:
    #         df[col + "_z"] = df.groupby("Ticker")[col].transform(
    #             lambda x: (x - x.rolling(20, min_periods=5).mean()) / x.rolling(20, min_periods=5).std()
    #         )

    # ====================================================
    # 4️⃣ Encode Regime Columns
    # ====================================================
    regime_mapping = {"Bearish": -1, "Neutral": 0, "Bullish": 1, "High Volatility": 2}

    for col in ["NIFTY_Enhanced_Regime_for_RD", "BANKNIFTY_Enhanced_Regime_for_RD"]:
        if col in df.columns:
            # Map all strings first
            df[col] = df[col].map(regime_mapping)
            # Fill remaining NaNs (originally missing or unmapped labels)
            df[col] = df[col].fillna(0).astype(int)

    # ====================================================
    # 5️⃣ Optional Smoothing for Noisy Derived Ratios
    # ====================================================
    if smooth:
        smooth_cols = [
            "MACD_diff", "EMA_diff", "VWAP_diff",
            "RSI_diff_NIFTY", "RSI_diff_BANKNIFTY",
            "ATR_vs_NIFTY", "ATR_vs_BANKNIFTY"
        ]
        for col in smooth_cols:
            if col in df.columns:
                df[col] = df.groupby("Ticker")[col].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # ====================================================
    # 6️⃣ Trim NaNs at Start and End per Ticker
    # ====================================================
    critical_cols = [c for c in df.columns if "_rel" in c or "_z" in c or c in ["ATR_14", "Range_pct"]]
    trimmed = []
    for ticker, group in df.groupby("Ticker", group_keys=False):
        valid_mask = group[critical_cols].notna().all(axis=1)
        if valid_mask.any():
            first_valid_idx = valid_mask.idxmax()
            group = group.loc[first_valid_idx:]
        trimmed.append(group)
    df = pd.concat(trimmed, ignore_index=True)

    # Trim tail NaNs in target
    trimmed_tail = []
    for ticker, group in df.groupby("Ticker", group_keys=False):
        valid_mask = group["future_return"].notna()
        if valid_mask.any():
            last_valid_idx = valid_mask[::-1].idxmax()
            group = group.loc[:last_valid_idx]
        trimmed_tail.append(group)
    df = pd.concat(trimmed_tail, ignore_index=True)

    # ====================================================
    # 7️⃣ Select ML Columns (Core + Derived)
    # ====================================================
    ml_cols = [
        "Date", "Ticker", "future_return",
        # Price relatives
        "EMA_diff", "EMA_20_rel", "EMA_50_rel", "VWAP_rel",
        "BB_mid_20_rel", "BB_upper_20_rel", "BB_lower_20_rel",
        "NIFTY_EMA_20_rel", "NIFTY_EMA_50_rel",
        "BANKNIFTY_EMA_20_rel", "BANKNIFTY_EMA_50_rel",
        # Derived metrics
        "MACD_diff", "RSI_diff_NIFTY", "RSI_diff_BANKNIFTY",
        "VWAP_diff", "VWAP_NIFTY_diff", "VWAP_BANKNIFTY_diff",
        "BB_Width", "BB_Pos", "NIFTY_BB_Width", "BANKNIFTY_BB_Width",
        "Close_vs_EMA20", "Close_vs_EMA50", "BB_Mid_Dev",
        "RSI_vs_MACD", "RSI_vs_NIFTY", "RSI_vs_BANKNIFTY",
        "Range_over_ATR", "Range_over_Close", "VWAP_minus_Close",
        "VWAP_minus_EMA20", "ATR_vs_NIFTY", "ATR_vs_BANKNIFTY",
        "log_ATR_vs_NIFTY",
        # # Lag features
        # "Range_pct_lag1", "Range_pct_lag2", "Range_pct_lag3",
        # "RSI_14_lag1", "RSI_14_lag2", "RSI_14_lag3",
        # "MACD_lag1", "MACD_lag2", "MACD_lag3",
        # "ATR_14_lag1", "ATR_14_lag2", "ATR_14_lag3",
        # lag requires previous data, which we do not have at cuurent
        "Range_pct", "RSI_14", "MACD", "ATR_14"
        # Z-scores
        # "RSI_14_z", "MACD_z", "MACD_signal_z", "%K_z", "%D_z",
        # "NIFTY_RSI_14_z", "NIFTY_MACD_z", "NIFTY_MACD_signal_z",
        # "BANKNIFTY_RSI_14_z", "BANKNIFTY_MACD_z", "BANKNIFTY_MACD_signal_z",
        # Volumes
        "Volume_log", "Cum_Vol_log", "Cum_TPV_log",
        "NIFTY_Volume_log", "NIFTY_Cum_Vol_log",
        "BANKNIFTY_Volume_log", "BANKNIFTY_Cum_Vol_log",
        # Regime
        "NIFTY_Enhanced_Regime_for_RD", "BANKNIFTY_Enhanced_Regime_for_RD"
    ]
    df_ml = df[[c for c in ml_cols if c in df.columns]].copy()

    # ====================================================
    # 8️⃣ Forward/Backward Fill per Ticker (safe)
    # ====================================================
    numeric_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
    for ticker, group in df_ml.groupby("Ticker", group_keys=False):
        df_ml.loc[group.index, numeric_cols] = (
            group[numeric_cols]
            .fillna(method="ffill", limit=1)
            .fillna(method="bfill", limit=1)
        )

    # ====================================================
    # 9️⃣ Final Global Standardization (critical)
    # ====================================================
    # Columns that should NEVER be scaled
    do_not_scale = [
        "future_return", 
        "NIFTY_Enhanced_Regime_for_RD",
        "BANKNIFTY_Enhanced_Regime_for_RD"
    ]

    # Only scale numeric columns except excluded ones
    scale_cols = [c for c in df_ml.select_dtypes(include=[np.number]).columns 
                if c not in do_not_scale]

    if scale:
        scaler = StandardScaler()
        df_ml[scale_cols] = scaler.fit_transform(df_ml[scale_cols])

    # ====================================================
    # Done
    # ====================================================
    print("Preprocessing complete.")
    print(f"Final ML columns: {len(df_ml.columns)} | Shape: {df_ml.shape}")
    return df_ml


# ========================================
# 3️⃣ Script Entry Point
# ========================================
if __name__ == "__main__":
    input_folder = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/data/features_ready.csv"
    output_path = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/data/normalized_data_for_ml.csv"

    print("Loading derived feature CSVs...")
    df = pd.read_csv(input_folder)

    print("Preprocessing derived features...")
    df_features = preprocess_derived_features(df)

    print(f"Saving preprocessed ML-ready dataset to {output_path} ...")
    df_features.to_csv(output_path, index=False)
    print("Done.")
