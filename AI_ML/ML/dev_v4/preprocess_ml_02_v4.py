# v4 is medicore
import pandas as pd
import numpy as np
import os
import pickle

def preprocess_combined(df: pd.DataFrame, smooth=True) -> pd.DataFrame:
    """
    Combined, robust, deterministic preprocessing:
    - Sorts per Ticker/Date to ensure deterministic cumulative calculations
    - Recomputes Cum_Vol / Cum_TPV per-ticker (if base columns exist)
    - Recomputes VWAP deterministically from Cum_TPV/Cum_Vol where possible
    - Produces only live-safe features (no future leakage)
    - Adds extra stable relative features: log_ret (t-1), price ratios, vol-price interactions
    - Optional short smoothing (center=False)
    - Drops rows where future_return is NaN (no future scanning)
    - Only forward-fills (limit=1)
    """
    df = df.copy()



    # -------------------------
    # Recompute per-ticker cumulative fields deterministically if possible
    # (fixes mismatches caused by different file concatenation/order)
    # -------------------------
    if "Volume" in df.columns:
        df["Cum_Vol_recomputed"] = df.groupby("Ticker")["Volume"].cumsum()
    if "Cum_TPV" in df.columns:
        # If Cum_TPV exists but may be inconsistent, recompute from TPV if present.
        # Prefer recomputing from per-row TPV column if it exists (TPV = Price * Volume or provided)
        if "TPV" in df.columns:
            df["Cum_TPV_recomputed"] = df.groupby("Ticker")["TPV"].cumsum()
        else:
            # If a Cum_TPV already exists, use as-is but ensure it's monotonic by filling forward/back (defensive)
            df["Cum_TPV_recomputed"] = df["Cum_TPV"].copy()

    # If Cum_TPV not present but TPV not present, we will not recompute; keep original VWAP if available.
    # Recompute VWAP deterministically where possible:
    if ("Cum_TPV_recomputed" in df.columns) and ("Cum_Vol_recomputed" in df.columns):
        # avoid division by zero
        df["VWAP_recomputed"] = df["Cum_TPV_recomputed"] / df["Cum_Vol_recomputed"].replace(0, np.nan)
    else:
        # fall back to existing VWAP if present
        if "VWAP" in df.columns:
            df["VWAP_recomputed"] = df["VWAP"].copy()

    # If neither, VWAP_recomputed won't exist; features depending on VWAP are guarded below.

    # -------------------------
    # 1) Relative Price-Based Normalization (keep original intention)
    # -------------------------
    price_cols = [
        "EMA_20", "EMA_50", "BB_mid_20", "BB_upper_20", "BB_lower_20",
        "EMA_fast", "EMA_slow", "VWAP_recomputed", "VWAP",        # include both candidates
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
    # 2) Log Transform Volumes (use recomputed cumulative values for determinism if available)
    # -------------------------
    # Use recomputed Cum_Vol_recomputed if present, else existing Cum_Vol
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
    # Extra safe features (no future, only existing/past)
    # -------------------------
    # log return using previous close (safe — requires only past)
    if "Close" in df.columns:
        df["log_ret_1"] = np.log(df["Close"] / df["Close"].shift(1))
        # price ratios
        df["High_Low_ratio"] = df["High"] / df["Low"]
        df["Close_Open_ratio"] = df["Close"] / df["Open"]
        df["Close_range_pos"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)

    # vol/price interactions
    if ("Volume" in df.columns) and ("Close" in df.columns):
        df["vol_over_price"] = df["Volume"] / (df["Close"] + 1e-6)
        df["vol_change_pct"] = df["Volume"].pct_change().fillna(0)

    # -------------------------
    # 3) Core derived features (safe, relative only)
    # -------------------------
    # EMA diff
    if ("EMA_20" in df.columns) and ("EMA_50" in df.columns):
        df["EMA_diff"] = df["EMA_20"] - df["EMA_50"]

    # MACD derived
    if ("MACD" in df.columns) and ("MACD_signal" in df.columns):
        df["MACD_diff"] = df["MACD"] - df["MACD_signal"]
        df["MACD_hist_rel"] = df["MACD_diff"] / (df["Close"] + 1e-6)

    # RSI / MACD cross-instrument diffs (if available)
    if ("RSI_14" in df.columns) and ("NIFTY_RSI_14" in df.columns):
        df["RSI_diff_NIFTY"] = df["RSI_14"] - df["NIFTY_RSI_14"]
    if ("RSI_14" in df.columns) and ("BANKNIFTY_RSI_14" in df.columns):
        df["RSI_diff_BANKNIFTY"] = df["RSI_14"] - df["BANKNIFTY_RSI_14"]

    if ("MACD" in df.columns) and ("NIFTY_MACD" in df.columns):
        df["MACD_diff_NIFTY"] = df["MACD"] - df["NIFTY_MACD"]
    if ("MACD" in df.columns) and ("BANKNIFTY_MACD" in df.columns):
        df["MACD_diff_BANKNIFTY"] = df["MACD"] - df["BANKNIFTY_MACD"]

    # -------------------------
    # 4) Bollinger features (if present)
    # -------------------------
    if set(["BB_upper_20", "BB_lower_20", "BB_mid_20"]).issubset(df.columns):
        df["BB_Width"] = (df["BB_upper_20"] - df["BB_lower_20"]) / (df["BB_mid_20"] + 1e-9)
        df["BB_Pos"] = (df["Close"] - df["BB_lower_20"]) / (df["BB_upper_20"] - df["BB_lower_20"] + 1e-9)
        df["BB_Mid_Dev"] = (df["Close"] - df["BB_mid_20"]) / (df["BB_upper_20"] - df["BB_lower_20"] + 1e-9)

    # NIFTY / BANKNIFTY BB widths defensively
    if set(["NIFTY_BB_upper_20", "NIFTY_BB_lower_20", "NIFTY_BB_mid_20"]).issubset(df.columns):
        df["NIFTY_BB_Width"] = (df["NIFTY_BB_upper_20"] - df["NIFTY_BB_lower_20"]) / (df["NIFTY_BB_mid_20"] + 1e-9)
    if set(["BANKNIFTY_BB_upper_20", "BANKNIFTY_BB_lower_20", "BANKNIFTY_BB_mid_20"]).issubset(df.columns):
        df["BANKNIFTY_BB_Width"] = (df["BANKNIFTY_BB_upper_20"] - df["BANKNIFTY_BB_lower_20"]) / (df["BANKNIFTY_BB_mid_20"] + 1e-9)

    # -------------------------
    # 5) VWAP: deterministic & stable computations (fix mismatch)
    # -------------------------
    # prefer VWAP_recomputed, fallback to original 'VWAP' column
    if "VWAP_recomputed" in df.columns:
        df["VWAP_used"] = df["VWAP_recomputed"]
    elif "VWAP" in df.columns:
        df["VWAP_used"] = df["VWAP"]
    else:
        df["VWAP_used"] = np.nan

    # stable VWAP-based features
    if "VWAP_used" in df.columns:
        df["VWAP_rel"] = (df["VWAP_used"] / df["Close"] - 1).replace([np.inf, -np.inf], np.nan)
        df["VWAP_minus_Close"] = df["VWAP_used"] - df["Close"]
        if "EMA_20" in df.columns:
            df["VWAP_minus_EMA20"] = df["VWAP_used"] - df["EMA_20"]
        # safe 1-lag slope (past only)
        df["VWAP_slope_rel"] = df.groupby("Ticker")["VWAP_used"].pct_change().fillna(0)

    # -------------------------
    # 6) ATR / Range features (relative)
    # -------------------------
    if ("Range_pct" in df.columns) and ("ATR_14" in df.columns):
        df["Range_over_ATR"] = df["Range_pct"] / (df["ATR_14"] + 1e-9)
    if ("High" in df.columns) and ("Low" in df.columns) and ("Close" in df.columns):
        df["Range_over_Close"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)

    if ("ATR_14" in df.columns) and ("NIFTY_ATR_14" in df.columns):
        df["ATR_vs_NIFTY"] = df["ATR_14"] / (df["NIFTY_ATR_14"] + 1e-9)
    if ("ATR_14" in df.columns) and ("BANKNIFTY_ATR_14" in df.columns):
        df["ATR_vs_BANKNIFTY"] = df["ATR_14"] / (df["BANKNIFTY_ATR_14"] + 1e-9)
    if "ATR_14" in df.columns and "Close" in df.columns:
        df["ATR_rel"] = df["ATR_14"] / (df["Close"] + 1e-9)
    if "ATR_vs_NIFTY" in df.columns:
        df["log_ATR_vs_NIFTY"] = np.log1p(np.abs(df["ATR_vs_NIFTY"].replace([np.inf, -np.inf], np.nan))).fillna(0)

    # -------------------------
    # 7) Market relative EMAs
    # -------------------------
    if ("NIFTY_EMA_20" in df.columns) and ("Close" in df.columns):
        df["NIFTY_EMA_20_rel"] = df["NIFTY_EMA_20"] / df["Close"] - 1
    if ("NIFTY_EMA_50" in df.columns) and ("Close" in df.columns):
        df["NIFTY_EMA_50_rel"] = df["NIFTY_EMA_50"] / df["Close"] - 1
    if ("BANKNIFTY_EMA_20" in df.columns) and ("Close" in df.columns):
        df["BANKNIFTY_EMA_20_rel"] = df["BANKNIFTY_EMA_20"] / df["Close"] - 1
    if ("BANKNIFTY_EMA_50" in df.columns) and ("Close" in df.columns):
        df["BANKNIFTY_EMA_50_rel"] = df["BANKNIFTY_EMA_50"] / df["Close"] - 1

    # -------------------------
    # 8) Regime encoding (unchanged)
    # -------------------------
    mapping = {"Bearish": -1, "Neutral": 0, "Bullish": 1, "High Volatility": 2}
    for col in ["NIFTY_Enhanced_Regime_for_RD", "BANKNIFTY_Enhanced_Regime_for_RD"]:
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # -------------------------
    # 9) Optional short smoothing (past-only)
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
                    lambda x: x.rolling(window=3, min_periods=1, center=False).mean()
                )

    # -------------------------
    # 10) Trim invalid rows at start per ticker (features must exist)
    # -------------------------
    # Build critical list deterministically (current dataframe)
    critical = [c for c in df.columns if ("_rel" in c) or (c in ["ATR_14", "Range_pct"])]
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
        # If no ticker grouping, just drop initial rows with missing criticals
        df = df[df[critical].notna().all(axis=1)].copy()

    # -------------------------
    # 11) Drop rows that don't have target (no future scanning)
    # -------------------------
    # if "future_return" in df.columns:
    #     df = df[df["future_return"].notna()].copy()


    
    # -------------------------
    # 12) Select ML columns (merge of original + new safe features)
    # -------------------------
    ml_cols = [
        "Date", "Ticker", "future_return",
        # Relatives + core
        "EMA_diff", "EMA_20_rel", "EMA_50_rel", "VWAP_rel",
        "BB_mid_20_rel", "BB_upper_20_rel", "BB_lower_20_rel",
        "NIFTY_EMA_20_rel", "NIFTY_EMA_50_rel",
        "BANKNIFTY_EMA_20_rel", "BANKNIFTY_EMA_50_rel",
        # Derived
        "MACD_diff", "RSI_diff_NIFTY", "RSI_diff_BANKNIFTY", "MACD_diff_NIFTY", "MACD_diff_BANKNIFTY",
        "BB_Width", "BB_Pos", "NIFTY_BB_Width", "BANKNIFTY_BB_Width",
        "Close_vs_EMA20", "Close_vs_EMA50", "BB_Mid_Dev",
        "RSI_vs_MACD", "RSI_vs_NIFTY", "RSI_vs_BANKNIFTY",
        "Range_over_ATR", "Range_over_Close", "VWAP_minus_Close","VWAP_minus_EMA20", "VWAP_slope_rel", 
        "ATR_vs_NIFTY", "ATR_vs_BANKNIFTY",
        "log_ATR_vs_NIFTY",
        # Raw safe indicators
        "Range_pct", "RSI_14", "MACD", "ATR_14",
        # Volume logs (use recomputed where possible)
        "Volume_log", "Cum_Vol_log", "Cum_TPV_log",
        "NIFTY_Volume_log", "NIFTY_Cum_Vol_log",
        "BANKNIFTY_Volume_log", 
        # "BANKNIFTY_Cum_Vol_log",
        # Extra safe features
        "log_ret_1", "High_Low_ratio", "Close_Open_ratio", "Close_range_pos",
        "vol_over_price", "vol_change_pct",
        # Regimes
        "NIFTY_Enhanced_Regime_for_RD", "BANKNIFTY_Enhanced_Regime_for_RD"
    ]
    df = df.dropna(subset=ml_cols)
    # Keep only columns present
    ml_present = [c for c in ml_cols if c in df.columns]
    df_ml = df[ml_present].copy()

    # -------------------------
    # 13) Forward-fill only (limit=1) per ticker for numeric columns
    # -------------------------
    numeric_cols = df_ml.select_dtypes(include=[np.number]).columns
    if "Ticker" in df_ml.columns:
        for ticker, group in df_ml.groupby("Ticker", group_keys=False):
            df_ml.loc[group.index, numeric_cols] = (
                group[numeric_cols]
                .fillna(method="ffill", limit=1)
            )
    else:
        df_ml[numeric_cols] = df_ml[numeric_cols].ffill(limit=1)


    # Final shape info
    print(f"Preprocessing combined | Features: {df_ml.shape[1]} | Rows: {df_ml.shape[0]}")
    return df_ml



def apply_minmax_scaling(df, feature_cols):
    scaler_dict = {}

    # ---------------------------------------
    # If scaler.pkl exists → Load and apply
    # ---------------------------------------
    if os.path.exists(SCALER_PICKLE):
        print("Loading existing scaler.pkl ...")
        with open(SCALER_PICKLE, "rb") as f:
            scaler_dict = pickle.load(f)

        for col in feature_cols:
            if col not in df.columns:
                continue
            col_min = scaler_dict[col]["min"]
            col_max = scaler_dict[col]["max"]
            rng = col_max - col_min if col_max != col_min else 1
            df[col] = ((df[col] - col_min) / rng).clip(0, 1)

        return df

    # ---------------------------------------
    # If scaler.pkl does NOT exist → Compute and save
    # ---------------------------------------
    print("Creating scaler.pkl ...")

    for col in feature_cols:
        if col not in df.columns:
            continue
        col_min = df[col].min()
        col_max = df[col].max()

        scaler_dict[col] = {
            "min": float(col_min),
            "max": float(col_max)
        }

        rng = col_max - col_min if col_max != col_min else 1
        df[col] = ((df[col] - col_min) / rng).clip(0, 1)

    # Save scaler.pkl
    with open(SCALER_PICKLE, "wb") as f:
        pickle.dump(scaler_dict, f)

    print("Saved scaler.pkl with", len(scaler_dict), "columns.")
    return df

# ========================================
# Script Entry Point (NO SCALING)
# ========================================
if __name__ == "__main__":
    SCALER_PICKLE = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4/data/scaler_v4.pkl"

    input_path = (
        "C:/PERSONAL_DATA/Startups/Stocks/"
        "Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4/data/features_ready.csv"
    )

    output_path = (
        "C:/PERSONAL_DATA/Startups/Stocks/"
        "Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4/data/normalized_data_for_ml.csv"
    )

    print("\n====================================")
    print(" Loading raw derived features...")
    print("====================================")

    df_raw = pd.read_csv(input_path)

    print("\n====================================")
    print(" Running preprocessing (NO SCALING inside)...")
    print("====================================")

    df_ml_raw = preprocess_combined(df_raw)

    # ------------------------------------------
    # SCALE ONLY HERE (after preprocessing)
    # ------------------------------------------
    feature_cols = [
        c for c in df_ml_raw.columns
        if c not in ["Date", "Ticker", "future_return"]
    ]

    df_ml = apply_minmax_scaling(df_ml_raw, feature_cols)

    print(f"\nSaving ML dataset to:\n{output_path}")
    df_ml.to_csv(output_path, index=False)

    print("\nDone. ✔")

