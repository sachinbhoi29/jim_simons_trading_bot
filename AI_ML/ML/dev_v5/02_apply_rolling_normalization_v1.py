import pandas as pd
import numpy as np
import pickle
import os

# ======================================================
# Dataset Builder V5  (CLEAN + STRICT + SAFE)
# ======================================================

def load_normalized_data(normalized_csv_path, ml_cols_pickle):
    print("Loading normalized ML data...")
    df = pd.read_csv(normalized_csv_path)

    print("Loading ML column order (from preprocessor)...")
    with open(ml_cols_pickle, "rb") as f:
        ml_cols = pickle.load(f)

    # Ensure strict ordering
    ml_cols = [c for c in ml_cols if c in df.columns]
    df = df[ml_cols]

    print(f"Loaded normalized ML dataset | Rows={df.shape[0]} | Cols={df.shape[1]}")
    return df, ml_cols


# ======================================================
# Build ML sequences
# ======================================================

def build_sequences(df, ml_cols, seq_len=60, target_col="future_return"):
    print(f"Building sequences | seq_len = {seq_len}")

    feature_cols = [
        c for c in ml_cols
        if c not in ["Date", "Ticker", target_col]
    ]

    X_list, y_list, tickers, dates = [], [], [], []

    for ticker, g in df.groupby("Ticker"):
        g = g.reset_index(drop=True)

        for i in range(len(g) - seq_len):
            seq = g.loc[i:i + seq_len - 1, feature_cols].values
            target = g.loc[i + seq_len, target_col]

            X_list.append(seq)
            y_list.append(target)
            tickers.append(ticker)
            dates.append(g.loc[i + seq_len, "Date"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"Sequences built | X shape = {X.shape} | y shape = {y.shape}")
    return X, y, tickers, dates, feature_cols


# ======================================================
# Time-based Split
# ======================================================

def train_val_test_split(X, y, tickers, dates, train_ratio=0.70, val_ratio=0.15):

    total = len(X)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"Split completed:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ======================================================
# Save dataset parts
# ======================================================

def save_dataset(output_dir, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)

    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)

    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    with open(os.path.join(output_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    print("Dataset saved successfully.")


# ======================================================
# MAIN ENTRY
# ======================================================

if __name__ == "__main__":

    # *** dev_v5 paths ONLY ***
    NORMALIZED_CSV = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v5/data/normalized_data_for_ml.csv"
    ML_COLS_PICKLE = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v5/data/preprocessor_v5_ml_cols.pkl"
    OUTPUT_DATASET_DIR = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v5/datasets"

    # LOAD
    df, ml_cols = load_normalized_data(NORMALIZED_CSV, ML_COLS_PICKLE)

    # BUILD SEQUENCES
    X, y, tickers, dates, feature_cols = build_sequences(
        df,
        ml_cols,
        seq_len=60,
        target_col="future_return"
    )

    # SPLIT
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(X, y, tickers, dates)

    # SAVE
    save_dataset(
        OUTPUT_DATASET_DIR,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_cols
    )
