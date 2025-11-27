import pandas as pd
import numpy as np
import os
import pickle

# ==========================
# V5 Preprocessor (v1-compatible + extras)
# ==========================

def _safe_div(a, b):
    return a / b.replace(0, np.nan)

def _pct(a, b):
    return (a / b) - 1

def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def preprocess_combined_full_v5(df: pd.DataFrame, smooth=True) -> pd.DataFrame:
    df = df.copy()

    # ======================================================
    # Recompute Cum_Vol, Cum_TPV, VWAP (deterministic)
    # ======================================================
    if "Volume" in df.columns:
        if "Ticker" in df.columns:
            df["Cum_Vol_recomputed"] = df.groupby("Ticker")["Volume"].cumsum()
        else:
            df["Cum_Vol_recomputed"] = df["Volume"].cumsum()

    if "TPV" in df.columns:
        if "Ticker" in df.columns:
            df["Cum_TPV_recomputed"] = df.groupby("Ticker")["TPV"].cumsum()
        else:
            df["Cum_TPV_recomputed"] = df["TPV"].cumsum()

    if "Cum_Vol_recomputed" in df.columns and "Cum_TPV_recomputed" in df.columns:
        df["VWAP_recomputed"] = df["Cum_TPV_recomputed"] / df["Cum_Vol_recomputed"].replace(0, np.nan)
    elif "VWAP" in df.columns:
        df["VWAP_recomputed"] = df["VWAP"]

    # ======================================================
    # v1 Relatives (preserved)
    # ======================================================
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

    # ======================================================
    # Log Volumes (v1)
    # ======================================================
    vol_sources = {
        "Volume": "Volume",
        "Cum_Vol": "Cum_Vol_recomputed",
        "Cum_TPV": "Cum_TPV_recomputed",
        "NIFTY_Volume": "NIFTY_Volume",
        "NIFTY_Cum_Vol": "NIFTY_Cum_Vol",
        "BANKNIFTY_Volume": "BANKNIFTY_Volume",
        "BANKNIFTY_Cum_Vol": "BANKNIFTY_Cum_Vol",
    }
    for out_col, src in vol_sources.items():
        if src in df.columns:
            df[f"{out_col}_log"] = np.log1p(df[src].clip(lower=0))

    # ======================================================
    # Basic price/volume features (v1 preserved)
    # ======================================================
    if {"Close", "High", "Low", "Open"}.issubset(df.columns):
        df["log_ret_1"] = np.log(df["Close"] / df["Close"].shift(1))
        df["High_Low_ratio"] = df["High"] / df["Low"]
        df["Close_Open_ratio"] = df["Close"] / df["Open"]
        df["Close_range_pos"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)

    if "Volume" in df.columns:
        df["vol_over_price"] = df["Volume"] / (df["Close"] + 1e-6)
        df["vol_change_pct"] = df["Volume"].pct_change().fillna(0)

    # ======================================================
    # EMA / MACD relatives (v1 preserved)
    # ======================================================
    if {"EMA_20", "EMA_50"}.issubset(df.columns):
        df["EMA_diff"] = df["EMA_20"] - df["EMA_50"]
        df["Close_vs_EMA20"] = df["Close"] - df["EMA_20"]
        df["Close_vs_EMA50"] = df["Close"] - df["EMA_50"]

    if {"MACD", "MACD_signal"}.issubset(df.columns):
        df["MACD_diff"] = df["MACD"] - df["MACD_signal"]
        df["MACD_hist_rel"] = df["MACD_diff"] / (df["Close"] + 1e-6)

    # RSI relatives
    if "RSI_14" in df.columns:
        if "NIFTY_RSI_14" in df.columns:
            df["RSI_diff_NIFTY"] = df["RSI_14"] - df["NIFTY_RSI_14"]
            df["RSI_vs_NIFTY"] = df["RSI_14"] / (df["NIFTY_RSI_14"] + 1e-6)
        if "BANKNIFTY_RSI_14" in df.columns:
            df["RSI_diff_BANKNIFTY"] = df["RSI_14"] - df["BANKNIFTY_RSI_14"]
            df["RSI_vs_BANKNIFTY"] = df["RSI_14"] / (df["BANKNIFTY_RSI_14"] + 1e-6)

    # ======================================================
    # Bollinger (v1 preserved)
    # ======================================================
    if {"BB_upper_20", "BB_lower_20", "BB_mid_20"}.issubset(df.columns):
        up = df["BB_upper_20"]
        lo = df["BB_lower_20"]
        mid = df["BB_mid_20"]
        df["BB_Width"] = (up - lo) / (mid + 1e-9)
        df["BB_Pos"] = (df["Close"] - lo) / (up - lo + 1e-9)
        df["BB_Mid_Dev"] = (df["Close"] - mid) / (up - lo + 1e-9)

    # ======================================================
    # VWAP (v1 preserved)
    # ======================================================
    if "VWAP_recomputed" in df.columns:
        df["VWAP_used"] = df["VWAP_recomputed"]
    else:
        df["VWAP_used"] = df.get("VWAP", np.nan)

    df["VWAP_rel"] = (df["VWAP_used"] / df["Close"] - 1).replace([np.inf, -np.inf], np.nan)
    df["VWAP_minus_Close"] = df["VWAP_used"] - df["Close"]
    if "EMA_20" in df.columns:
        df["VWAP_minus_EMA20"] = df["VWAP_used"] - df["EMA_20"]

    if "Ticker" in df.columns:
        df["VWAP_slope_rel"] = df.groupby("Ticker")["VWAP_used"].pct_change().fillna(0)

    # ======================================================
    # ATR / Range (v1 preserved)
    # ======================================================
    if "ATR_14" in df.columns:
        df["ATR_rel"] = df["ATR_14"] / (df["Close"] + 1e-9)

    # ======================================================
    # Regimes (v1 preserved)
    # ======================================================
    mapping = {"Bearish": -1, "Neutral": 0, "Bullish": 1, "High Volatility": 2}
    for col in ["NIFTY_Enhanced_Regime_for_RD", "BANKNIFTY_Enhanced_Regime_for_RD"]:
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # ======================================================
    # v5 MICROSTRUCTURE ADD-ONS
    # ======================================================
    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        rng = (df["High"] - df["Low"]).replace(0, np.nan)
        body = (df["Close"] - df["Open"]).abs()
        df["candle_body_pct"] = (body / rng).fillna(0)
        df["wick_upper_pct"] = ((df["High"] - df[["Close", "Open"]].max(axis=1)) / rng).fillna(0)
        df["wick_lower_pct"] = ((df[["Close", "Open"]].min(axis=1) - df["Low"]) / rng).fillna(0)
        df["body_vs_range"] = df["Close"] - ((df["High"] + df["Low"]) / 2)

    # ======================================================
    # v5 Percentiles
    # ======================================================
    if "Volume" in df.columns:
        df["Volume_pctile_20"] = (
            df["Volume"].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        ).fillna(0)

    # ======================================================
    # v5 Stock vs Index relatives
    # ======================================================
    if {"Close", "NIFTY_Close"}.issubset(df.columns):
        df["Stock_vs_NIFTY_rel"] = df["Close"] / df["NIFTY_Close"] - 1

    if {"Close", "BANKNIFTY_Close"}.issubset(df.columns):
        df["Stock_vs_BANKNIFTY_rel"] = df["Close"] / df["BANKNIFTY_Close"] - 1

    # ======================================================
    # TRIM START (v1 logic)
    # ======================================================
    critical = [c for c in df.columns if "_rel" in c]
    cleaned = []

    if "Ticker" in df.columns:
        for ticker, g in df.groupby("Ticker", group_keys=False):
            mask = g[critical].notna().all(axis=1)
            if mask.any():
                first = mask.idxmax()
                cleaned.append(g.loc[first:])
        df = pd.concat(cleaned, ignore_index=True)

    # ======================================================
    # FIXED FORWARD FILL (NO DEPRECATION)
    # ======================================================
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if "Ticker" in df.columns:
        for ticker, g in df.groupby("Ticker", group_keys=False):
            df.loc[g.index, numeric_cols] = g[numeric_cols].ffill(limit=1)
    else:
        df[numeric_cols] = df[numeric_cols].ffill(limit=1)

    print(f"Preprocessing complete | Features: {df.shape[1]} | Rows: {df.shape[0]}")

    # ======================================================
    # v1 COLUMN LIST + v5 EXTRAS
    # ======================================================
    v1_ml_cols = [
        "Date", "Ticker", "future_return",
        "EMA_diff", "EMA_20_rel", "EMA_50_rel", "VWAP_rel",
        "BB_mid_20_rel", "BB_upper_20_rel", "BB_lower_20_rel",
        "NIFTY_EMA_20_rel", "NIFTY_EMA_50_rel",
        "BANKNIFTY_EMA_20_rel", "BANKNIFTY_EMA_50_rel",
        "Close_vs_EMA20", "Close_vs_EMA50",
        "MACD_diff", "MACD_diff_NIFTY", "MACD_diff_BANKNIFTY",
        "RSI_diff_NIFTY", "RSI_diff_BANKNIFTY",
        "RSI_vs_MACD", "RSI_vs_NIFTY", "RSI_vs_BANKNIFTY",
        "BB_Width", "BB_Pos", "BB_Mid_Dev",
        "NIFTY_BB_Width", "BANKNIFTY_BB_Width",
        "VWAP_minus_Close", "VWAP_minus_EMA20", "VWAP_slope_rel",
        "VWAP_NIFTY_diff", "VWAP_BANKNIFTY_diff",
        "Range_over_ATR", "Range_over_Close", "ATR_vs_NIFTY", "ATR_vs_BANKNIFTY",
        "ATR_14", "log_ATR_vs_NIFTY",
        "Range_pct", "RSI_14", "MACD",
        "Volume_log", "Cum_Vol_log", "Cum_TPV_log",
        "NIFTY_Volume_log", "NIFTY_Cum_Vol_log",
        "BANKNIFTY_Volume_log", "BANKNIFTY_Cum_Vol_log",
        "log_ret_1", "High_Low_ratio", "Close_Open_ratio", "Close_range_pos",
        "vol_over_price", "vol_change_pct",
        "NIFTY_Enhanced_Regime_for_RD", "BANKNIFTY_Enhanced_Regime_for_RD"
    ]

    v5_extra = [
        "candle_body_pct", "wick_upper_pct", "wick_lower_pct", "body_vs_range",
        "Volume_pctile_20",
        "Stock_vs_NIFTY_rel", "Stock_vs_BANKNIFTY_rel"
    ]

    ml_cols = v1_ml_cols + v5_extra
    ml_present = [c for c in ml_cols if c in df.columns]

    df_ml = df[ml_present].dropna(subset=[c for c in v1_ml_cols if c in df.columns])

    return df_ml, ml_present


# ==========================
# SCALER (v1-style)
# ==========================
def apply_minmax_scaling_v1_style(df, feature_cols, SCALER_PICKLE):
    scaler_dict = {}

    if os.path.exists(SCALER_PICKLE):
        print("Loading existing scaler.pkl ...")
        with open(SCALER_PICKLE, "rb") as f:
            scaler_dict = pickle.load(f)
    else:
        print("No existing scaler.pkl found. Creating new one...")

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Ensure numeric (fix for TypeError)
        df[col] = pd.to_numeric(df[col], errors="coerce")

        if col in scaler_dict:
            col_min = scaler_dict[col]["min"]
            col_max = scaler_dict[col]["max"]
        else:
            col_min = df[col].min()
            col_max = df[col].max()
            scaler_dict[col] = {"min": float(col_min), "max": float(col_max)}

        rng = col_max - col_min if col_max != col_min else 1
        df[col] = ((df[col] - col_min) / rng).clip(0, 1)

    with open(SCALER_PICKLE, "wb") as f:
        pickle.dump(scaler_dict, f)

    print(f"Scaling complete | Features scaled: {len(feature_cols)}")
    return df


# ==========================
# MAIN (script entry) - v1 style
# ==========================
if __name__ == "__main__":
    # Update these paths as needed
    SCALER_PICKLE = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4/data/preprocessor_v5_scaler.pkl"
    ML_COLS_PICKLE = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4/data/preprocessor_v5_ml_cols.pkl"
    input_path = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4/data/features_ready.csv"
    output_path = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4/data/normalized_data_for_ml.csv"

    print("Loading raw features...")
    df_raw = pd.read_csv(input_path)

    print("Running full preprocessing (v5)...")
    df_ml_raw, ml_present = preprocess_combined_full_v5(df_raw, smooth=True)

    # Save ML column list (exact order we will use)
    with open(ML_COLS_PICKLE, "wb") as f:
        pickle.dump(ml_present, f)
    print(f"Saved ML cols to {ML_COLS_PICKLE} | columns: {len(ml_present)}")

    # Identify numeric features to scale (exclude target)
    feature_cols = df_ml_raw.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in ["future_return"]]

    # Ensure feature_cols follow ml_present order (only numeric ones)
    feature_cols = [c for c in ml_present if c in feature_cols]

    df_ml = apply_minmax_scaling_v1_style(df_ml_raw, feature_cols, SCALER_PICKLE)

    print(f"Saving ML dataset to {output_path} ...")
    # Save only the ML-present columns (ordered)
    df_ml[ml_present].to_csv(output_path, index=False)

    print("Done. ✔")
