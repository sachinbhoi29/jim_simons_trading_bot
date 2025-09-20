# pip install pandas matplotlib openpyxl numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_options(csv_file):
    """
    Load CSV with Greeks and options data
    """
    df = pd.read_csv(csv_file)
    return df

def build_delta_neutral_strategy(df, positions, lot_size=50):
    """
    Build a delta-neutral option strategy.
    
    positions: list of dicts
        [{"type": "CE" or "PE", "strike": 25500, "position": -1}, ...]
        position=-1 means sell, +1 means buy
    """
    # Initialize columns
    df["Position"] = 0
    df["PnL"] = 0
    df["Net_Delta"] = 0
    df["Net_Gamma"] = 0
    df["Net_Vega"] = 0
    df["Net_Theta"] = 0
    
    # Filter relevant strikes and CE/PE
    for pos in positions:
        opt_type = pos["type"]
        strike = pos["strike"]
        position = pos["position"]
        lot = lot_size
        
        if opt_type == "CE":
            row = df[df["strikePrice"] == strike].copy()
            premium = row["Call_LTP"].values[0]
            delta = row["Call_Delta"].values[0]
            gamma = row["Call_Gamma"].values[0]
            vega = row["Call_Vega"].values[0]
            theta = row["Call_Theta_per_day"].values[0]
        else:
            row = df[df["strikePrice"] == strike].copy()
            premium = row["Put_LTP"].values[0]
            delta = row["Put_Delta"].values[0]
            gamma = row["Put_Gamma"].values[0]
            vega = row["Put_Vega"].values[0]
            theta = row["Put_Theta_per_day"].values[0]

        df.loc[df["strikePrice"] == strike, "Position"] = position * lot
        df.loc[df["strikePrice"] == strike, "PnL"] = position * lot * (premium)
        df.loc[df["strikePrice"] == strike, "Net_Delta"] = position * lot * delta
        df.loc[df["strikePrice"] == strike, "Net_Gamma"] = position * lot * gamma
        df.loc[df["strikePrice"] == strike, "Net_Vega"] = position * lot * vega
        df.loc[df["strikePrice"] == strike, "Net_Theta"] = position * lot * theta
        
    return df

def plot_strategy(df, spot_price, positions):
    """
    Plot payoff at expiry and Greeks
    """
    strikes = df["strikePrice"].values
    pnl = df["PnL"].values
    delta = df["Net_Delta"].values
    gamma = df["Net_Gamma"].values
    vega = df["Net_Vega"].values
    theta = df["Net_Theta"].values

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    # Payoff
    ax[0].bar(strikes, pnl, width=40, color='skyblue')
    ax[0].axvline(spot_price, color='red', linestyle='--', label="Spot Price")
    ax[0].set_title("Delta Neutral Strategy Payoff at Expiry")
    ax[0].set_xlabel("Strike Price")
    ax[0].set_ylabel("PnL")
    ax[0].legend()

    # Greeks
    ax[1].plot(strikes, delta, label="Delta")
    ax[1].plot(strikes, gamma, label="Gamma")
    ax[1].plot(strikes, vega, label="Vega")
    ax[1].plot(strikes, theta, label="Theta")
    ax[1].axhline(0, color='black', linestyle='--')
    ax[1].set_title("Net Greeks per Strike")
    ax[1].set_xlabel("Strike Price")
    ax[1].set_ylabel("Greek Value")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

# ---------------- Example Usage ----------------
csv_file = "feature_development/options/dev/NIFTY_options_07Oct2025_with_greeks.csv"
df = load_options(csv_file)

# Define a sample delta neutral strategy: sell ATM CE + sell ATM PE
spot_price = 25500
positions = [
    {"type": "CE", "strike": 25500, "position": -1},
    {"type": "PE", "strike": 25500, "position": -1}
]

df_strategy = build_delta_neutral_strategy(df, positions, lot_size=50)
plot_strategy(df_strategy, spot_price, positions)
