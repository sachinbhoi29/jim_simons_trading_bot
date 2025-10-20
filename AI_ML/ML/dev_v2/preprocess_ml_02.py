import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import glob
import os



# ========================================
# 2️⃣ Preprocess Derived Features
# ========================================
def preprocess_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a dataset of derived features:
    - Handles missing values safely per ticker
    - Normalizes numeric columns
    - Encodes categorical/regime columns
    - Returns ML-ready DataFrame
    """

    # -------------------------------
    # D. Normalize numeric columns
    # -------------------------------

    # 1️⃣ Normalize / create _rel columns for price indicators
    price_cols = ["EMA_20", "EMA_50", "BB_mid_20", "BB_upper_20", "BB_lower_20",
                "EMA_fast", "EMA_slow", "VWAP",
                "NIFTY_EMA_20", "NIFTY_EMA_50",
                "BANKNIFTY_EMA_20", "BANKNIFTY_EMA_50"]

    for col in price_cols:
        if col in df.columns:
            if "BANKNIFTY" in col:
                base_col = "BANKNIFTY_Close"
                df[col + "_rel"] = (df[col] / df[base_col]) - 1
            elif "NIFTY" in col:
                base_col = "NIFTY_Close"
                df[col + "_rel"] = (df[col] / df[base_col]) - 1
            else:
                base_col = "Close"
                df[col + "_rel"] = (df[col] / df[base_col]) - 1

    # 2️⃣ Log-transform volume-related columns
    volume_cols = ["Volume", "Cum_Vol", "Cum_TPV", "NIFTY_Volume", "NIFTY_Cum_Vol",
                "BANKNIFTY_Volume", "BANKNIFTY_Cum_Vol"]
    for col in volume_cols:
        if col in df.columns:
            df[col + "_log"] = np.log1p(df[col])

    # 3️⃣ Z-score normalize oscillators/momentum indicators
    zscore_cols = ["RSI_14", "MACD", "MACD_signal", "%K", "%D",
                "NIFTY_RSI_14", "NIFTY_MACD", "NIFTY_MACD_signal",
                "BANKNIFTY_RSI_14", "BANKNIFTY_MACD", "BANKNIFTY_MACD_signal"]

    for col in zscore_cols:
        if col in df.columns:
            mean_ = df[col].rolling(20, min_periods=5).mean()
            std_ = df[col].rolling(20, min_periods=5).std()
            df[col + "_z"] = (df[col] - mean_) / std_

    # -------------------------------
    #  C. Encode categorical/regime columns
    # -------------------------------
    regime_mapping = {"Bearish": -1,"Neutral": 0,"Bullish": 1,"High Volatility": 2}

    for col in ["NIFTY_Enhanced_Regime_for_RD", "BANKNIFTY_Enhanced_Regime_for_RD"]:
        if col in df.columns:
            # Replace NaN with 0
            df[col] = df[col].fillna(0)

            # Map categories (skip 0 since it's already numeric)
            df[col] = df[col].map(lambda x: regime_mapping.get(x, x) if isinstance(x, str) else x)

            # Check for any unmapped values
            if df[col].isna().any():
                invalid_vals = df.loc[df[col].isna(), col].unique()
                print(f"Unmapped / invalid values in column '{col}': {invalid_vals}")
                raise ValueError(f"Column '{col}' has unmapped or invalid categories: {invalid_vals}")

            df[col] = df[col].astype(int)


    # -------------------------------
    #  E. Trim initial NaNs per ticker for critical features
    # -------------------------------
    critical_cols = [c for c in df.columns if "_rel" in c or "_z" in c or c in ["ATR_14", "Range_pct"]]
    trimmed = []
    for ticker, group in df.groupby("Ticker", group_keys=False):
        valid_mask = group[critical_cols].notna().all(axis=1)
        if valid_mask.any():
            first_valid_idx = valid_mask.idxmax()
            group = group.loc[first_valid_idx:]
        trimmed.append(group)
    df = pd.concat(trimmed, ignore_index=True)

    # -------------------------------
    #  F. Trim tail NaNs for target
    # -------------------------------
    trimmed_tail = []
    for ticker, group in df.groupby("Ticker", group_keys=False):
        valid_mask = group["future_return"].notna()
        if valid_mask.any():
            last_valid_idx = valid_mask[::-1].idxmax()
            group = group.loc[:last_valid_idx]
        trimmed_tail.append(group)
    df = pd.concat(trimmed_tail, ignore_index=True)

    # -------------------------------
    #  G. Final ML Columns Selection
    # -------------------------------
    # All derived features + target + categorical regimes


    ml_cols = [
        "Date", "Ticker", "future_return",

        # -----------------------
        # Price relative features
        # -----------------------
        "EMA_diff_rel",
        "EMA_20_rel", "EMA_50_rel", "EMA_fast_rel", "EMA_slow_rel",
        "BB_mid_20_rel", "BB_upper_20_rel", "BB_lower_20_rel",
        "VWAP_rel",
        "NIFTY_EMA_20_rel", "NIFTY_EMA_50_rel",
        "BANKNIFTY_EMA_20_rel", "BANKNIFTY_EMA_50_rel",

        # -----------------------
        # Derived ratios & differences
        # -----------------------
        "MACD_diff",
        "RSI_diff_NIFTY", "RSI_diff_BANKNIFTY",
        "ATR_Range_ratio", "Volatility_ratio_Market", "Volatility_ratio_Bank",
        "VWAP_diff", "VWAP_NIFTY_diff", "VWAP_BANKNIFTY_diff",
        "Price_vs_NIFTY", "Price_vs_BANKNIFTY",
        "BB_Width", "BB_Pos", "NIFTY_BB_Width", "BANKNIFTY_BB_Width",
        "Close_vs_EMA20", "Close_vs_EMA50", "BB_Mid_Dev",
        "RSI_vs_MACD", "RSI_vs_NIFTY", "RSI_vs_BANKNIFTY",
        "Range_over_ATR", "Range_over_Close", "VWAP_minus_Close", "VWAP_minus_EMA20",
        "ATR_vs_NIFTY", "ATR_vs_BANKNIFTY", "log_ATR_vs_NIFTY",

        # -----------------------
        # Lagged features
        # -----------------------
        "Range_pct_lag1", "Range_pct_lag2", "Range_pct_lag3",
        "RSI_14_lag1", "RSI_14_lag2", "RSI_14_lag3",
        "MACD_lag1", "MACD_lag2", "MACD_lag3",
        "ATR_14_lag1", "ATR_14_lag2", "ATR_14_lag3",

        # -----------------------
        # Oscillators normalized
        # -----------------------
        "RSI_14_z", "MACD_z", "MACD_signal_z", "%K_z", "%D_z",
        "NIFTY_RSI_14_z", "NIFTY_MACD_z", "NIFTY_MACD_signal_z",
        "BANKNIFTY_RSI_14_z", "BANKNIFTY_MACD_z", "BANKNIFTY_MACD_signal_z",

        # -----------------------
        # Volume normalized
        # -----------------------
        "Volume_log", "Cum_Vol_log", "Cum_TPV_log",
        "NIFTY_Volume_log", "NIFTY_Cum_Vol_log",
        "BANKNIFTY_Volume_log", "BANKNIFTY_Cum_Vol_log",

        # -----------------------
        # Regime / categorical
        # -----------------------
        "NIFTY_Enhanced_Regime_for_RD", "BANKNIFTY_Enhanced_Regime_for_RD"
    ]

    df_ml = df[ml_cols].copy()
    # -------------------------------
    # B. Handle Missing Values (time-series safe)
    # -------------------------------
    # Select all numeric columns
    numeric_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()

    # Apply limited forward/backward fill per ticker
    for ticker, group in df_ml.groupby("Ticker", group_keys=False):
        df_ml.loc[group.index, numeric_cols] = (
            group[numeric_cols]
            .fillna(method="ffill", limit=1)
            .fillna(method="bfill", limit=1)
        )


    print("Preprocessing complete.")
    print("Columns used for ML:", df_ml.columns.tolist())
    print("Shape:", df_ml.shape)

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
    print("✅ Done.")
