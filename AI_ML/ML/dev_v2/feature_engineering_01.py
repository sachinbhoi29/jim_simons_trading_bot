import pandas as pd
import numpy as np
import glob
import os

# ========================================
# 1️⃣ Load & Concatenate Multiple CSV Files
# ========================================
def load_and_concat_csvs(folder_path: str) -> pd.DataFrame: 
    """Load all CSV files from a folder and concatenate them into a single DataFrame."""
    file_list = glob.glob(os.path.join(folder_path, "*.csv"))
    if not file_list:
        raise ValueError(f"No CSV files found in folder: {folder_path}")
    dfs = [pd.read_csv(file) for file in file_list]
    all_data = pd.concat(dfs, ignore_index=True)
    return all_data


# ========================================
# 2️⃣ Feature Engineering Function
# ========================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced derived and relational financial indicators
    from the base technical and market features.
    """


    # -------------------------------
    #  A. Trend & Momentum Relationships
    # -------------------------------
    df["EMA_diff"] = df["EMA_20"] - df["EMA_50"]
    df["NIFTY_EMA_diff"] = df["NIFTY_EMA_20"] - df["NIFTY_EMA_50"]
    df["BANKNIFTY_EMA_diff"] = df["BANKNIFTY_EMA_20"] - df["BANKNIFTY_EMA_50"]

    # Relative differences normalized to Close
    df["EMA_diff_rel"] = df["EMA_diff"] / df["Close"]
    df["MACD_diff"] = df["MACD"] - df["MACD_signal"]

    # Cross-instrument divergence
    df["RSI_diff_NIFTY"] = df["RSI_14"] - df["NIFTY_RSI_14"]
    df["RSI_diff_BANKNIFTY"] = df["RSI_14"] - df["BANKNIFTY_RSI_14"]
    df["MACD_diff_NIFTY"] = df["MACD"] - df["NIFTY_MACD"]
    df["MACD_diff_BANKNIFTY"] = df["MACD"] - df["BANKNIFTY_MACD"]

    # -------------------------------
    #  B. Volatility & Range Metrics
    # -------------------------------
    df["ATR_Range_ratio"] = df["ATR_14"] / (df["Range_pct"].abs() + 1e-6)
    df["Volatility_ratio_Market"] = df["ATR_14"] / (df["NIFTY_ATR_14"] + 1e-6)
    df["Volatility_ratio_Bank"] = df["ATR_14"] / (df["BANKNIFTY_ATR_14"] + 1e-6)

    # -------------------------------
    #  C. Market Relative Price Strength
    # -------------------------------
    df["VWAP_diff"] = df["VWAP"] - df["Close"]
    df["VWAP_NIFTY_diff"] = df["VWAP"] - df["NIFTY_EMA_20"]
    df["VWAP_BANKNIFTY_diff"] = df["VWAP"] - df["BANKNIFTY_EMA_20"]

    df["Price_vs_NIFTY"] = (df["Close"] / df["NIFTY_Close"]) - 1
    df["Price_vs_BANKNIFTY"] = (df["Close"] / df["BANKNIFTY_Close"]) - 1

    # -------------------------------
    #  D. Bollinger Band Dynamics
    # -------------------------------
    df["BB_Width"] = (df["BB_upper_20"] - df["BB_lower_20"]) / df["BB_mid_20"]
    df["BB_Pos"] = (df["Close"] - df["BB_lower_20"]) / (df["BB_upper_20"] - df["BB_lower_20"] + 1e-6)

    df["NIFTY_BB_Width"] = (df["NIFTY_BB_upper_20"] - df["NIFTY_BB_lower_20"]) / df["NIFTY_BB_mid_20"]
    df["BANKNIFTY_BB_Width"] = (df["BANKNIFTY_BB_upper_20"] - df["BANKNIFTY_BB_lower_20"]) / df["BANKNIFTY_BB_mid_20"]

    # -------------------------------
    #  E. Momentum Persistence (Lagged Features)
    # -------------------------------
    lag_cols = ["Range_pct", "RSI_14", "MACD", "ATR_14"]
    for col in lag_cols:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df.groupby("Ticker")[col].shift(lag)

    df['Close_vs_EMA20'] = (df['Close'] / df['EMA_20']) - 1
    df['Close_vs_EMA50'] = (df['Close'] / df['EMA_50']) - 1
    df['BB_Mid_Dev'] = (df['Close'] - df['BB_mid_20']) / (df['BB_upper_20'] - df['BB_lower_20'] + 1e-6)
    df['RSI_vs_MACD'] = df['RSI_14'] / (df['MACD'] + 1e-6)
    df['RSI_vs_NIFTY'] = df['RSI_14'] - df['NIFTY_RSI_14']
    df['RSI_vs_BANKNIFTY'] = df['RSI_14'] - df['BANKNIFTY_RSI_14']
    df['Range_over_ATR'] = (df['High'] - df['Low']) / (df['ATR_14'] + 1e-6)
    df['Range_over_Close'] = (df['High'] - df['Low']) / df['Close']
    df['VWAP_minus_Close'] = df['VWAP'] - df['Close']
    df['VWAP_minus_EMA20'] = df['VWAP'] - df['EMA_20']
    df['ATR_vs_NIFTY'] = df['ATR_14'] / (df['NIFTY_ATR_14'] + 1e-6)
    df['ATR_vs_BANKNIFTY'] = df['ATR_14'] / (df['BANKNIFTY_ATR_14'] + 1e-6)
    df['log_ATR_vs_NIFTY'] = np.log1p(df['ATR_vs_NIFTY'])

    # -------------------------------
    #  H. Clean and Output
    # -------------------------------
    print("Feature engineering complete.")
    print("Shape:", df.shape)
    return df


# ========================================
# 3️⃣ Script Entry Point
# ========================================
if __name__ == "__main__":
    input_folder = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/data/input_csvs"
    output_path = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/data/features_ready.csv"

    print("Loading raw CSVs...")
    df = load_and_concat_csvs(input_folder)

    print("Engineering advanced features...")
    df_features = engineer_features(df)

    print(f"Saving enhanced dataset to {output_path} ...")
    df_features.to_csv(output_path, index=False)
    print("Done.")
