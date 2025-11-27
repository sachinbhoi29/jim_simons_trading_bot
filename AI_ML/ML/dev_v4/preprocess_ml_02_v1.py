# !!!!!!!!!!!!!!!!!!! better on 5d and worse on 1d
import pandas as pd
import numpy as np
import os
import pickle

# ========================================
# 1️⃣ Combined Preprocessing Function
# ========================================
def preprocess_combined_full(df: pd.DataFrame, smooth=True) -> pd.DataFrame:
    """
    Full preprocessing:
    - Combines old features + deterministic cumulative calculations
    - Future-safe only (no future leakage)
    - Optional short smoothing
    - Drops rows with NaNs in critical features
    """
    df = df.copy()

    # -------------------------
    # Sort per Ticker / Date
    # -------------------------
    if "Ticker" in df.columns and "Date" in df.columns:
        df.sort_values(["Ticker", "Date"], inplace=True)

    # -------------------------
    # Recompute Cum_Vol / Cum_TPV / VWAP deterministically
    # -------------------------
    if "Volume" in df.columns:
        df["Cum_Vol_recomputed"] = df.groupby("Ticker")["Volume"].cumsum()
    if "TPV" in df.columns:
        df["Cum_TPV_recomputed"] = df.groupby("Ticker")["TPV"].cumsum()
    elif "Cum_TPV" in df.columns:
        df["Cum_TPV_recomputed"] = df["Cum_TPV"].copy()

    if "Cum_Vol_recomputed" in df.columns and "Cum_TPV_recomputed" in df.columns:
        df["VWAP_recomputed"] = df["Cum_TPV_recomputed"] / df["Cum_Vol_recomputed"].replace(0, np.nan)
    elif "VWAP" in df.columns:
        df["VWAP_recomputed"] = df["VWAP"].copy()

    # -------------------------
    # 1) Relative Price Features
    # -------------------------
    price_cols = [
        "EMA_20", "EMA_50", "BB_mid_20", "BB_upper_20", "BB_lower_20",
        "EMA_fast", "EMA_slow", "VWAP_recomputed",
        "NIFTY_EMA_20", "NIFTY_EMA_50",
        "BANKNIFTY_EMA_20", "BANKNIFTY_EMA_50"
    ]
    for col in price_cols:
        if col in df.columns:
            if "BANKNIFTY" in col:
                base = "BANKNIFTY_Close"
            elif "NIFTY" in col:
                base = "NIFTY_Close"
            else:
                base = "Close"
            if base in df.columns:
                df[f"{col}_rel"] = (df[col] / df[base]) - 1

    # -------------------------
    # 2) Log Volumes
    # -------------------------
    vol_sources = {
        "Volume": "Volume",
        "Cum_Vol": "Cum_Vol_recomputed" if "Cum_Vol_recomputed" in df.columns else "Cum_Vol",
        "Cum_TPV": "Cum_TPV_recomputed" if "Cum_TPV_recomputed" in df.columns else "Cum_TPV",
        "NIFTY_Volume": "NIFTY_Volume",
        "NIFTY_Cum_Vol": "NIFTY_Cum_Vol",
        "BANKNIFTY_Volume": "BANKNIFTY_Volume",
        "BANKNIFTY_Cum_Vol": "BANKNIFTY_Cum_Vol",
    }
    for out_col, src in vol_sources.items():
        if src in df.columns:
            df[f"{out_col}_log"] = np.log1p(df[src].clip(lower=0))

    # -------------------------
    # 3) Extra safe features
    # -------------------------
    if "Close" in df.columns:
        df["log_ret_1"] = np.log(df["Close"] / df["Close"].shift(1))
        df["High_Low_ratio"] = df["High"] / df["Low"]
        df["Close_Open_ratio"] = df["Close"] / df["Open"]
        df["Close_range_pos"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)

    if "Volume" in df.columns and "Close" in df.columns:
        df["vol_over_price"] = df["Volume"] / (df["Close"] + 1e-6)
        df["vol_change_pct"] = df["Volume"].pct_change().fillna(0)

    # -------------------------
    # 4) Core derived features
    # -------------------------
    if "EMA_20" in df.columns and "EMA_50" in df.columns:
        df["EMA_diff"] = df["EMA_20"] - df["EMA_50"]
        df["Close_vs_EMA20"] = df["Close"] - df["EMA_20"]
        df["Close_vs_EMA50"] = df["Close"] - df["EMA_50"]

    if "MACD" in df.columns and "MACD_signal" in df.columns:
        df["MACD_diff"] = df["MACD"] - df["MACD_signal"]
        df["MACD_hist_rel"] = df["MACD_diff"] / (df["Close"] + 1e-6)

    # RSI / MACD cross diffs
    if "RSI_14" in df.columns:
        if "NIFTY_RSI_14" in df.columns:
            df["RSI_diff_NIFTY"] = df["RSI_14"] - df["NIFTY_RSI_14"]
            df["RSI_vs_NIFTY"] = df["RSI_14"] / (df["NIFTY_RSI_14"] + 1e-6)
        if "BANKNIFTY_RSI_14" in df.columns:
            df["RSI_diff_BANKNIFTY"] = df["RSI_14"] - df["BANKNIFTY_RSI_14"]
            df["RSI_vs_BANKNIFTY"] = df["RSI_14"] / (df["BANKNIFTY_RSI_14"] + 1e-6)
    if "MACD" in df.columns:
        if "NIFTY_MACD" in df.columns:
            df["MACD_diff_NIFTY"] = df["MACD"] - df["NIFTY_MACD"]
        if "BANKNIFTY_MACD" in df.columns:
            df["MACD_diff_BANKNIFTY"] = df["MACD"] - df["BANKNIFTY_MACD"]
    if "RSI_14" in df.columns and "MACD" in df.columns:
        df["RSI_vs_MACD"] = df["RSI_14"] / (df["MACD"] + 1e-6)

    # -------------------------
    # 5) Bollinger Features
    # -------------------------
    if set(["BB_upper_20", "BB_lower_20", "BB_mid_20"]).issubset(df.columns):
        df["BB_Width"] = (df["BB_upper_20"] - df["BB_lower_20"]) / (df["BB_mid_20"] + 1e-9)
        df["BB_Pos"] = (df["Close"] - df["BB_lower_20"]) / (df["BB_upper_20"] - df["BB_lower_20"] + 1e-9)
        df["BB_Mid_Dev"] = (df["Close"] - df["BB_mid_20"]) / (df["BB_upper_20"] - df["BB_lower_20"] + 1e-9)
    if set(["NIFTY_BB_upper_20", "NIFTY_BB_lower_20", "NIFTY_BB_mid_20"]).issubset(df.columns):
        df["NIFTY_BB_Width"] = (df["NIFTY_BB_upper_20"] - df["NIFTY_BB_lower_20"]) / (df["NIFTY_BB_mid_20"] + 1e-9)
    if set(["BANKNIFTY_BB_upper_20", "BANKNIFTY_BB_lower_20", "BANKNIFTY_BB_mid_20"]).issubset(df.columns):
        df["BANKNIFTY_BB_Width"] = (df["BANKNIFTY_BB_upper_20"] - df["BANKNIFTY_BB_lower_20"]) / (df["BANKNIFTY_BB_mid_20"] + 1e-9)

    # -------------------------
    # 6) VWAP Features
    # -------------------------
    if "VWAP_recomputed" in df.columns:
        df["VWAP_used"] = df["VWAP_recomputed"]
    elif "VWAP" in df.columns:
        df["VWAP_used"] = df["VWAP"]
    else:
        df["VWAP_used"] = np.nan

    if "VWAP_used" in df.columns:
        df["VWAP_rel"] = (df["VWAP_used"] / df["Close"] - 1).replace([np.inf, -np.inf], np.nan)
        df["VWAP_minus_Close"] = df["VWAP_used"] - df["Close"]
        if "EMA_20" in df.columns:
            df["VWAP_minus_EMA20"] = df["VWAP_used"] - df["EMA_20"]
        df["VWAP_slope_rel"] = df.groupby("Ticker")["VWAP_used"].pct_change().fillna(0)

    # Add market-relative VWAP differences
    if "VWAP_used" in df.columns:
        if "NIFTY_VWAP" in df.columns:
            df["VWAP_NIFTY_diff"] = df["VWAP_used"] - df["NIFTY_VWAP"]
        if "BANKNIFTY_VWAP" in df.columns:
            df["VWAP_BANKNIFTY_diff"] = df["VWAP_used"] - df["BANKNIFTY_VWAP"]

    # -------------------------
    # 7) ATR / Range Features
    # -------------------------
    if ("Range_pct" in df.columns) and ("ATR_14" in df.columns):
        df["Range_over_ATR"] = df["Range_pct"] / (df["ATR_14"] + 1e-9)
    if ("High" in df.columns) and ("Low" in df.columns) and ("Close" in df.columns):
        df["Range_over_Close"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)

    if "ATR_14" in df.columns:
        if "NIFTY_ATR_14" in df.columns:
            df["ATR_vs_NIFTY"] = df["ATR_14"] / (df["NIFTY_ATR_14"] + 1e-9)
        if "BANKNIFTY_ATR_14" in df.columns:
            df["ATR_vs_BANKNIFTY"] = df["ATR_14"] / (df["BANKNIFTY_ATR_14"] + 1e-9)
        df["ATR_rel"] = df["ATR_14"] / (df["Close"] + 1e-9)
    if "ATR_vs_NIFTY" in df.columns:
        df["log_ATR_vs_NIFTY"] = np.log1p(np.abs(df["ATR_vs_NIFTY"].replace([np.inf, -np.inf], np.nan))).fillna(0)

    # -------------------------
    # 8) Regimes
    # -------------------------
    mapping = {"Bearish": -1, "Neutral": 0, "Bullish": 1, "High Volatility": 2}
    for col in ["NIFTY_Enhanced_Regime_for_RD", "BANKNIFTY_Enhanced_Regime_for_RD"]:
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # -------------------------
    # 9) Optional smoothing
    # -------------------------
    if smooth:
        smooth_cols = [
            "MACD_diff", "EMA_diff", "VWAP_slope_rel",
            "RSI_diff_NIFTY", "RSI_diff_BANKNIFTY",
            "ATR_vs_NIFTY", "ATR_vs_BANKNIFTY"
        ]
        for col in smooth_cols:
            if col in df.columns:
                df[col] = df.groupby("Ticker")[col].transform(
                    lambda x: x.rolling(3, min_periods=1, center=False).mean()
                )

    # -------------------------
    # 10) Trim NaNs at start per ticker
    # -------------------------
    critical = [c for c in df.columns if "_rel" in c or c in ["ATR_14", "Range_pct"]]
    cleaned = []
    if "Ticker" in df.columns:
        for ticker, group in df.groupby("Ticker", group_keys=False):
            mask = group[critical].notna().all(axis=1)
            if mask.any():
                first = mask.idxmax()
                group = group.loc[first:]
            cleaned.append(group)
        df = pd.concat(cleaned, ignore_index=True)
    else:
        df = df[df[critical].notna().all(axis=1)].copy()

    # -------------------------
    # 11) Forward-fill numeric columns (limit=1)
    # -------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if "Ticker" in df.columns:
        for ticker, group in df.groupby("Ticker", group_keys=False):
            df.loc[group.index, numeric_cols] = group[numeric_cols].fillna(method="ffill", limit=1)
    else:
        df[numeric_cols] = df[numeric_cols].ffill(limit=1)

    print(f"Preprocessing complete | Features: {df.shape[1]} | Rows: {df.shape[0]}")

    ml_cols = [
            "Date", "Ticker", "future_return",
            # Price relatives & core
            "EMA_diff", "EMA_20_rel", "EMA_50_rel", "VWAP_rel",
            "BB_mid_20_rel", "BB_upper_20_rel", "BB_lower_20_rel",
            "NIFTY_EMA_20_rel", "NIFTY_EMA_50_rel",
            "BANKNIFTY_EMA_20_rel", "BANKNIFTY_EMA_50_rel",
            "Close_vs_EMA20", "Close_vs_EMA50",
            # Derived metrics
            "MACD_diff", "MACD_diff_NIFTY", "MACD_diff_BANKNIFTY",
            "RSI_diff_NIFTY", "RSI_diff_BANKNIFTY",
            "RSI_vs_MACD", "RSI_vs_NIFTY", "RSI_vs_BANKNIFTY",
            # Bollinger features
            "BB_Width", "BB_Pos", "BB_Mid_Dev",
            "NIFTY_BB_Width", "BANKNIFTY_BB_Width",
            # VWAP-based features
            "VWAP_minus_Close", "VWAP_minus_EMA20", "VWAP_slope_rel",
            "VWAP_NIFTY_diff", "VWAP_BANKNIFTY_diff",
            # ATR / Range
            "Range_over_ATR", "Range_over_Close", "ATR_vs_NIFTY", "ATR_vs_BANKNIFTY", "ATR_14", "log_ATR_vs_NIFTY",
            # Core indicators
            "Range_pct", "RSI_14", "MACD",
            # Volume logs
            "Volume_log", "Cum_Vol_log", "Cum_TPV_log",
            "NIFTY_Volume_log", "NIFTY_Cum_Vol_log",
            "BANKNIFTY_Volume_log", "BANKNIFTY_Cum_Vol_log",
            # Extra safe features
            "log_ret_1", "High_Low_ratio", "Close_Open_ratio", "Close_range_pos",
            "vol_over_price", "vol_change_pct",
            # Regimes
            "NIFTY_Enhanced_Regime_for_RD", "BANKNIFTY_Enhanced_Regime_for_RD"
        ]

    df_ml = df[[c for c in ml_cols if c in df.columns]].copy()
    df_ml = df_ml.dropna(subset=ml_cols)
    return df_ml

# ========================================
# 2️⃣ Scaling Function (Pickle-based)
# ========================================
def apply_minmax_scaling(df, feature_cols, SCALER_PICKLE):
    scaler_dict = {}

    # Load existing scaler if it exists
    if os.path.exists(SCALER_PICKLE):
        print("Loading existing scaler.pkl ...")
        with open(SCALER_PICKLE, "rb") as f:
            scaler_dict = pickle.load(f)
    else:
        print("No existing scaler.pkl found. Creating new one...")

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Ensure numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # If column exists in scaler_dict, use min/max from pickle
        if col in scaler_dict:
            col_min = scaler_dict[col]["min"]
            col_max = scaler_dict[col]["max"]
        else:
            # Compute min/max for new columns and add to pickle
            col_min = df[col].min()
            col_max = df[col].max()
            scaler_dict[col] = {"min": float(col_min), "max": float(col_max)}

        rng = col_max - col_min if col_max != col_min else 1
        df[col] = ((df[col] - col_min) / rng).clip(0, 1)

    # Save (or update) scaler.pkl
    with open(SCALER_PICKLE, "wb") as f:
        pickle.dump(scaler_dict, f)

    print(f"Scaling complete | Features scaled: {len(feature_cols)}")
    return df

# ========================================
# 3️⃣ Script Entry
# ========================================
if __name__ == "__main__":
    SCALER_PICKLE = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4/data/scaler_v1.pkl"  # Update path as needed
    input_path = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4/data/features_ready.csv"  # Update path as needed
    output_path = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4/data/normalized_data_for_ml.csv"  # Update path as needed

    print("Loading raw features...")
    df_raw = pd.read_csv(input_path)

    print("Running full preprocessing...")
    df_ml_raw = preprocess_combined_full(df_raw, smooth=True)

    # Identify numeric features to scale
    # feature_cols = [c for c in df_ml_raw.columns if c not in ["Date", "Ticker", "future_return"]]
    # Only numeric features (exclude target)
    feature_cols = df_ml_raw.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in ["future_return"]]

    df_ml = apply_minmax_scaling(df_ml_raw, feature_cols, SCALER_PICKLE)

    print(f"Saving ML dataset to {output_path} ...")
    df_ml.to_csv(output_path, index=False)
    print("Done. ✔")
