import pandas as pd
import numpy as np
import os
import pickle

# ======================================================
# 1️⃣ Preprocessor V5
# ======================================================

def preprocess_v5(
    input_csv,
    output_raw_csv,
    output_normalized_csv,
    scaler_pickle,
    ml_cols_pickle
):
    # ----------------------------
    # Load raw features
    # ----------------------------
    df = pd.read_csv(input_csv)
    print(f"Loaded raw features | Rows={df.shape[0]} | Cols={df.shape[1]}")

    # ----------------------------
    # Identify numeric columns (exclude non-numeric + target + identifiers)
    # ----------------------------
    exclude_cols = ["Date", "Ticker", "target"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # ----------------------------
    # Save intermediate CSV before normalization
    # ----------------------------
    df[["Date","Ticker","target"] + feature_cols].to_csv(output_raw_csv, index=False)
    print(f"Intermediate CSV saved (pre-normalization) → {output_raw_csv}")

    # ----------------------------
    # Apply min-max normalization (safe)
    # ----------------------------
    scaler_dict = {}
    df_scaled = df.copy()
    for col in feature_cols:
        col_min = df_scaled[col].min()
        col_max = df_scaled[col].max()
        rng = col_max - col_min if col_max != col_min else 1.0
        df_scaled[col] = ((df_scaled[col] - col_min) / rng).clip(0, 1)
        scaler_dict[col] = {"min": float(col_min), "max": float(col_max)}

    # ----------------------------
    # Save normalized CSV
    # ----------------------------
    df_scaled[["Date","Ticker","target"] + feature_cols].to_csv(output_normalized_csv, index=False)
    print(f"Normalized CSV saved → {output_normalized_csv}")

    # ----------------------------
    # Save scaler pickle
    # ----------------------------
    with open(scaler_pickle, "wb") as f:
        pickle.dump(scaler_dict, f)
    print(f"Scaler pickle saved → {scaler_pickle}")

    # ----------------------------
    # Save ML column list pickle (ordered)
    # ----------------------------
    ml_cols = ["Date", "Ticker", "target"] + feature_cols
    with open(ml_cols_pickle, "wb") as f:
        pickle.dump(ml_cols, f)
    print(f"ML columns pickle saved → {ml_cols_pickle}")

    return df_scaled, scaler_dict, ml_cols


# ======================================================
# MAIN ENTRY
# ======================================================
if __name__ == "__main__":
    # INPUT / OUTPUT PATHS (dev_v5)
    INPUT_CSV = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v5/data/features_raw.csv"
    INTERMEDIATE_CSV = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v5/data/features_before_norm.csv"
    NORMALIZED_CSV = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v5/data/normalized_data_for_ml.csv"
    SCALER_PICKLE = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v5/data/preprocessor_v5_scaler.pkl"
    ML_COLS_PICKLE = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v5/data/preprocessor_v5_ml_cols.pkl"

    preprocess_v5(INPUT_CSV, INTERMEDIATE_CSV, NORMALIZED_CSV, SCALER_PICKLE, ML_COLS_PICKLE)
