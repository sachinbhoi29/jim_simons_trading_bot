import pandas as pd
import numpy as np

# ===============================
# Load ML Data
# ===============================
path = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4_test/data/normalized_data_for_ml.csv"
df = pd.read_csv(path)
print("Loaded:", df.shape, "\n")

# ==========================================================
# Expected Ranges (custom per your feature list)
# ==========================================================
expected_ranges = {
    # Relative features typically between -5 and +5 max
    "_rel": (-5, 5),
    "EMA_diff": (-1000, 1000),
    "MACD_diff": (-1000, 1000),
    "MACD_diff_": (-1000, 1000),
    "BB_Width": (0, 10),
    "BB_Pos": (0, 1),
    "BB_Mid_Dev": (-10, 10),

    "RSI_14": (0, 100),
    "RSI_diff": (-100, 100),
    "RSI_vs_": (-100, 100),

    "MACD": (-5000, 5000),

    "Range_pct": (0, 2),   # 200% range max
    "Range_over_ATR": (0, 20),
    "Range_over_Close": (0, 1),

    "ATR_14": (0, 5000),
    "ATR_vs_NIFTY": (0, 20),
    "ATR_vs_BANKNIFTY": (0, 20),
    "log_ATR_vs_NIFTY": (-5, 5),

    "Volume_log": (0, 20),
    "_Vol_log": (0, 25),

    "Close_range_pos": (0, 1),
    "vol_change_pct": (-10, 10),
    "High_Low_ratio": (1, 5),
    "Close_Open_ratio": (0.2, 5),

    "MACD_diff_NIFTY": (-2000, 2000),
    "MACD_diff_BANKNIFTY": (-2000, 2000),

    "future_return": (-1, 1),

    # Regime columns
    "Enhanced_Regime_for_RD": (-2, 3),
}

# ==========================================================
# Helper to find expected range for a column name
# ==========================================================
def find_expected_range(col):
    for key, bounds in expected_ranges.items():
        if key in col:
            return bounds
    return None  # if no rule, skip range test

# ==========================================================
# Main validation
# ==========================================================
bad_columns = []

for col in df.columns:
    if df[col].dtype not in [float, int]:
        continue  # only numeric columns

    s = df[col]

    print(f"\nChecking column: {col}")

    # ------------------------------
    # 1. Check NaN
    # ------------------------------
    if s.isna().any():
        bad_columns.append(col)
        print(f"  ❌ NaN count = {s.isna().sum()}")
        print(df.loc[s.isna(), [col]].head())

    # ------------------------------
    # 2. Check INF
    # ------------------------------
    if np.isinf(s).any():
        bad_columns.append(col)
        print("  ❌ INF detected")
        print(df.loc[np.isinf(s), [col]].head())

    # ------------------------------
    # 3. Extremely large values
    # ------------------------------
    if (s.abs() > 1e12).any():
        bad_columns.append(col)
        print("  ❌ Extremely large values (> 1e12)")
        print(df.loc[s.abs() > 1e12, [col]].head())

    # ------------------------------
    # 4. Range validation (custom)
    # ------------------------------
    rng = find_expected_range(col)
    if rng:
        low, high = rng
        mask = (s < low) | (s > high)
        if mask.any():
            bad_columns.append(col)
            print(f"  ❌ Out of expected bounds {low}–{high} count = {mask.sum()}")
            print(df.loc[mask, [col]].head())

print("\n===================================================")
print("FINAL SUMMARY: COLUMNS WITH ANY PROBLEMS")
print("===================================================")

if bad_columns:
    for c in sorted(set(bad_columns)):
        print(" -", c)
else:
    print("✔ Dataset clean — no issues detected!")

print("\nDone.")
