import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import glob
import os


def load_and_concat_csvs(folder_path: str) -> pd.DataFrame: 
    """ Load all CSV files from a folder and concatenate them into a single DataFrame. Args: folder_path (str): Path to the folder containing CSV files. Returns: pd.DataFrame: Concatenated DataFrame. """ 
    # Find all CSV files in the folder 
    file_list = glob.glob(os.path.join(folder_path, "*.csv")) 
    if not file_list: 
        raise ValueError(f"No CSV files found in folder: {folder_path}") 
    # Read and collect all CSVs 
    dfs = [pd.read_csv(file) for file in file_list] 
    # Concatenate into a single DataFrame 
    all_data = pd.concat(dfs, ignore_index=True) 
    return all_data

def preprocess_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a multi-feature financial dataset for ML.
    Steps:
    1. Parse dates & sort
    2. Select key features
    3. Handle missing values (time-series safe)
    4. Normalize/scaling of numeric features
    5. Encode categorical columns
    6. Return cleaned DataFrame ready for model training
    """

    # -------------------------------
    # 2️⃣ Select Important Columns
    # -------------------------------
    # Main price and volume features
    price_cols = ["Close", "Open", "High", "Low", "Adj Close", "Range_pct", "future_return"]
    vol_cols = ["Volume", "Cum_Vol", "Cum_TPV", "VWAP","NIFTY_Volume","NIFTY_Cum_Vol","BANKNIFTY_Volume","BANKNIFTY_Cum_Vol"]

    # Technical indicators (main ticker)
    tech_cols = [
        "EMA_20", "EMA_50", "BB_mid_20", "BB_upper_20", "BB_lower_20",
        "ATR_14", "RSI_14", "EMA_fast", "EMA_slow", "MACD", "MACD_signal",
        "%K", "%D"
    ]

    # Market context (NIFTY and BANKNIFTY)
    nifty_cols = [
        "NIFTY_Close", "NIFTY_EMA_20", "NIFTY_EMA_50",
        "NIFTY_RSI_14", "NIFTY_MACD", "NIFTY_MACD_signal"
    ]
    banknifty_cols = [
        "BANKNIFTY_Close", "BANKNIFTY_EMA_20", "BANKNIFTY_EMA_50",
        "BANKNIFTY_RSI_14", "BANKNIFTY_MACD", "BANKNIFTY_MACD_signal"
    ]

    keep_cols = (
        ["Date", "Ticker"] +
        price_cols + vol_cols + tech_cols +
        nifty_cols + banknifty_cols 
    )

    df = df[[c for c in keep_cols if c in df.columns]].copy()


    # -------------------------------
    # 4️⃣ Feature Normalization (Explicit)
    # -------------------------------
    df['future_return'] = df['future_return'] / 100  
    # ---------- 1) Price-based indicators (% deviation from Close)
    price_rel_cols = [
        "EMA_20", "EMA_50", "BB_mid_20", "BB_upper_20", "BB_lower_20",
        "EMA_fast", "EMA_slow", "VWAP",
        "NIFTY_EMA_20", "NIFTY_EMA_50",
        "BANKNIFTY_EMA_20", "BANKNIFTY_EMA_50"
    ]
    for col in price_rel_cols:
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
        # # Sanity check Print sample calculations
        # counter = 0
        # max_samples = 5  # show first 5 samples
        # if counter < max_samples:
        #     print(f"Column: {col}")
        #     print("Raw values:", df[col].values[:max_samples])
        #     print("Base values:", df[base_col].values[:max_samples])
        #     print("Relative:", df[col + "_rel"].values[:max_samples])
        #     print("-" * 40)
        #     counter += 1

    # ---------- 2) Volume columns (log scale)
    volume_cols = ["Volume", "Cum_Vol", "Cum_TPV","NIFTY_Volume","NIFTY_Cum_Vol","BANKNIFTY_Volume","BANKNIFTY_Cum_Vol"]
    for col in volume_cols:
        if col in df.columns:
            df[col + "_log"] = np.log1p(df[col])

    # ---------- 3) Volatility metrics relative to Close
    if "ATR_14" in df.columns:
        df["ATR_rel"] = df["ATR_14"] / df["Close"]
    if "NIFTY_ATR_14" in df.columns:
        df["NIFTY_ATR_rel"] = df["NIFTY_ATR_14"] / df["NIFTY_Close"]
    if "BANKNIFTY_ATR_14" in df.columns:
        df["BANKNIFTY_ATR_rel"] = df["BANKNIFTY_ATR_14"] / df["BANKNIFTY_Close"]

    # ---------- 4) Index relative strength (Close vs EMA_20)
    if "NIFTY_Close" in df.columns and "NIFTY_EMA_20" in df.columns:
        df["NIFTY_rel"] = (df["NIFTY_Close"] / df["NIFTY_EMA_20"]) - 1
    if "BANKNIFTY_Close" in df.columns and "BANKNIFTY_EMA_20" in df.columns:
        df["BANKNIFTY_rel"] = (df["BANKNIFTY_Close"] / df["BANKNIFTY_EMA_20"]) - 1

    # ---------- 5) Z-score normalization for oscillators & momentum
    zscore_cols = [
        "RSI_14", "MACD", "MACD_signal", "%K", "%D",
        "NIFTY_RSI_14", "NIFTY_MACD", "NIFTY_MACD_signal",
        "BANKNIFTY_RSI_14", "BANKNIFTY_MACD", "BANKNIFTY_MACD_signal"
    ]

    for col in zscore_cols:
        if col in df.columns:
            mean_ = df[col].rolling(20, min_periods=5).mean()
            std_ = df[col].rolling(20, min_periods=5).std()
            df[col + "_z"] = (df[col] - mean_) / std_
            
            # # ----- debug: print first 5 or 10 calculations -----
            # print(f"\nColumn: {col}")
            # for i in range(min(10, len(df))):  # first 10 rows
            #     if not pd.isna(df[col + "_z"].iloc[i]):
            #         print(f"Index: {i}, Value: {df[col].iloc[i]:.4f}, "
            #             f"Rolling mean: {mean_.iloc[i]:.4f}, Std: {std_.iloc[i]:.4f}, "
            #             f"Z-score: {df[col + '_z'].iloc[i]:.4f}")


    # -------------------------------
    # 5️⃣ Encode categorical features
    # -------------------------------
    regime_mapping = {
        "Bearish": -1,
        "Neutral": 0,
        "Bullish": 1,
        "High Volatility": 2
    }

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
    # 8️⃣ Optional Standardization for all numeric features
    # -------------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])


    # -------------------------------
    # 3️⃣ Handle Missing Values
    # -------------------------------
    # Forward fill then back fill (time-series safe)

    # Define columns that are safe to fill
    fill_cols = [
        "EMA_20", "EMA_50", "BB_mid_20", "BB_upper_20", "BB_lower_20",
        "RSI_14", "ATR_14", "MACD", "MACD_signal",
        "EMA_fast", "EMA_slow", "%K", "%D",
        "NIFTY_EMA_20", "NIFTY_EMA_50", "NIFTY_RSI_14", "NIFTY_MACD", "NIFTY_MACD_signal",
        "BANKNIFTY_EMA_20", "BANKNIFTY_EMA_50", "BANKNIFTY_RSI_14", "BANKNIFTY_MACD", "BANKNIFTY_MACD_signal",
        "NIFTY_EMA_20_rel","NIFTY_EMA_50_rel","BANKNIFTY_EMA_20_rel","BANKNIFTY_EMA_50_rel",
        "NIFTY_Volume_log","NIFTY_Cum_Vol_log","BANKNIFTY_Volume_log",
        "NIFTY_rel","BANKNIFTY_rel","BANKNIFTY_Cum_Vol_log",
        "NIFTY_RSI_14_z","NIFTY_MACD_z","NIFTY_MACD_signal_z","BANKNIFTY_RSI_14_z",
        "BANKNIFTY_MACD_z","BANKNIFTY_MACD_signal_z"

    ]
    # Apply limited forward/backward fill only on those columns
    df[fill_cols] = (
        df[fill_cols]
        .fillna(method="ffill", limit=1)
        .fillna(method="bfill", limit=1)
    )


    # -------------------------------
    # 7️⃣b Trim initial NaN patch per Ticker
    # -------------------------------
    critical_cols = [
        "BB_mid_20", "BB_upper_20", "BB_lower_20", "ATR_14", "RSI_14",
        "%K", "%D",
        "NIFTY_Close", "NIFTY_EMA_20", "NIFTY_EMA_50", "NIFTY_RSI_14",
        "NIFTY_MACD", "NIFTY_MACD_signal",
        "BANKNIFTY_Close", "BANKNIFTY_EMA_20", "BANKNIFTY_EMA_50",
        "BANKNIFTY_RSI_14", "BANKNIFTY_MACD", "BANKNIFTY_MACD_signal",
        # normalized/derived versions
        "BB_mid_20_rel", "BB_upper_20_rel", "BB_lower_20_rel",
        "ATR_rel", "NIFTY_rel", "BANKNIFTY_rel",
        "RSI_14_z", "MACD_z", "MACD_signal_z", "%K_z", "%D_z",'BANKNIFTY_RSI_14_z',
        "NIFTY_EMA_20_rel","NIFTY_EMA_50_rel","BANKNIFTY_EMA_20_rel","BANKNIFTY_EMA_50_rel",
        "NIFTY_Volume_log","NIFTY_Cum_Vol_log","BANKNIFTY_Volume_log",
        "NIFTY_rel","BANKNIFTY_rel"]

    if "Ticker" in df.columns:
        trimmed = []
        for ticker, group in df.groupby("Ticker", group_keys=False):
            # find first valid index across all critical columns
            valid_mask = group[critical_cols].notna().all(axis=1)
            if valid_mask.any():
                first_valid_idx = valid_mask.idxmax()
                group = group.loc[first_valid_idx:]
            trimmed.append(group)

        df = pd.concat(trimmed, ignore_index=True)

    # -------------------------------
    # 7️⃣c Trim tail NaN patch for 'future_return' per Ticker
    # -------------------------------
    if "Ticker" in df.columns and "future_return" in df.columns:
        trimmed_tail = []
        for ticker, group in df.groupby("Ticker", group_keys=False):
            # find last valid index where future_return is not NaN
            valid_mask = group["future_return"].notna()
            if valid_mask.any():
                last_valid_idx = valid_mask[::-1].idxmax()  # last valid row
                group = group.loc[:last_valid_idx]
            trimmed_tail.append(group)
        df = pd.concat(trimmed_tail, ignore_index=True)


    ml_cols = [
        # Target + basic features
        "Date", "Ticker", "future_return", "Range_pct",
        # Categorical / regime
        "Fibo_Status_Last_Close", 
        # Normalized price indicators (main ticker)
        "EMA_20_rel", "EMA_50_rel", "BB_mid_20_rel", "BB_upper_20_rel", "BB_lower_20_rel",
        # "EMA_fast_rel", "EMA_slow_rel", 
        "VWAP_rel",
        # Normalized price indicators (market context)
        "NIFTY_EMA_20_rel", "NIFTY_EMA_50_rel",
        "BANKNIFTY_EMA_20_rel", "BANKNIFTY_EMA_50_rel",
        # Normalized volume features
        "Volume_log", "Cum_Vol_log", "Cum_TPV_log",
        "NIFTY_Volume_log","NIFTY_Cum_Vol_log","BANKNIFTY_Volume_log","BANKNIFTY_Cum_Vol_log",
        # Volatility indicators
        "ATR_rel", 
        # Market relative strength
        "NIFTY_rel", "BANKNIFTY_rel",
        # Z-score normalized oscillators / momentum (main ticker)
        "RSI_14_z", "MACD_z", "MACD_signal_z", "%K_z", "%D_z",   
        # Z-score normalized oscillators / momentum (market context)
        "NIFTY_RSI_14_z", "NIFTY_MACD_z", "NIFTY_MACD_signal_z",
        "BANKNIFTY_RSI_14_z", "BANKNIFTY_MACD_z", "BANKNIFTY_MACD_signal_z"
    ]
    # Filter DataFrame
    df_ml = df[[c for c in ml_cols if c in df.columns]].copy()
    print("Columns used for ML:", df_ml.columns.tolist())
    print("Shape:", df_ml.shape)

    return df_ml


if __name__ == "__main__":
    # Load your dataset
    df = load_and_concat_csvs('C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev/data/input_csvs')
    # Preprocess
    clean_df = preprocess_financial_data(df)
    clean_df.to_csv("AI_ML/ML/dev/data/clean_df.csv",index=False)
    print(clean_df.head())
    print(clean_df.shape)