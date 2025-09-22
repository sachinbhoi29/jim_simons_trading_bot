import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --- Load Excel ---
file_path = "feature_development/options/dev/NIFTY_options_07Oct2025.xlsx"
df = pd.read_excel(file_path)

# --- Spot price ---
spot = df['CE_underlyingValue'].iloc[0]

# --- Find nearest strikes above and below spot ---
strikes = df['StrikePrice'].sort_values().unique()

# Generate equidistant pairs relative to spot
distances = np.arange(0, max(strikes)-min(strikes)+50, 50)  # every 50 points
pairs = []

for d in distances:
    call_strike = spot + d
    put_strike = spot - d

    # Find nearest available strike in the data
    call_row = df.iloc[(df['StrikePrice'] - call_strike).abs().argsort()[:1]]
    put_row = df.iloc[(df['StrikePrice'] - put_strike).abs().argsort()[:1]]

    if not call_row.empty and not put_row.empty:
        call_ltp = call_row['Call_LTP'].values[0]
        put_ltp = put_row['Put_LTP'].values[0]

        # Calculate probabilities
        total = call_ltp + put_ltp
        prob_above = call_ltp / total if total > 0 else np.nan  # Bullish
        prob_below = put_ltp / total if total > 0 else np.nan   # Bearish
        prob_above = prob_above*100
        prob_below = prob_below*100

        pairs.append({
            'Spot': spot,
            'Call_Strike': call_row['StrikePrice'].values[0],
            'Put_Strike': put_row['StrikePrice'].values[0],
            'Call_LTP': call_ltp,
            'Put_LTP': put_ltp,
            'Prob_Above': prob_above,
            'Prob_Below': prob_below,
            'Distance_from_Spot': d
        })

# --- Convert to DataFrame and save ---
pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv("feature_development/options/dev/NIFTY_Equidistant_ITM_Probabilities.csv", index=False)
print("CSV saved: NIFTY_Equidistant_ITM_Probabilities.csv")
print(pairs_df)


