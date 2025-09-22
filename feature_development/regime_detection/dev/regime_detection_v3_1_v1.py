import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ta  # Using 'ta' instead of 'pandas_ta'
import yfinance as yf

# Step 1: Load data
# --- Example usage ---
# Define NIFTY50 ticker symbol
ticker = "^NSEI"

# Download 1 year of daily data
df = yf.download(ticker, period="2y", interval="1d",multi_level_index=False)

# Optional: Reset index if needed
df.reset_index(inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# --- Step 2: Compute Technical Indicators using 'ta' ---
df["50EMA"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
df["200EMA"] = ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator()
df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
df["ATR_50"] = df["ATR"].rolling(50).mean()  # Rolling ATR mean for volatility

macd_obj = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
df["MACD"] = macd_obj.macd()
df["Signal"] = macd_obj.macd_signal()
df["MACD_Hist"] = df["MACD"] - df["Signal"]  # MACD Histogram for momentum

bb_obj = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
df["BB_Upper"] = bb_obj.bollinger_hband()
df["BB_Lower"] = bb_obj.bollinger_lband()

# --- Step 3: Define Enhanced Market Regime Classification ---
def detect_enhanced_regime(row, prev_regime, prev_count, df, index):
    """Improved market regime classification"""
    # Last 3 days of MACD_Hist for momentum confirmation
    if index >= 3:
        macd_hist_3d = df["MACD_Hist"].iloc[index-2:index+1].values
        three_day_macd_neg = all(macd < 0 for macd in macd_hist_3d)
    else:
        three_day_macd_neg = False

    # --- Bullish Regime ---
    if row["50EMA"] > row["200EMA"] and row["MACD"] > row["Signal"] and row["RSI"] > 55:
        return "Bullish"

    # --- Bearish Regime (Earlier Detection) ---
    elif (
        row["50EMA"] < row["200EMA"] 
        and row["MACD"] < row["Signal"]
        and row["RSI"] < 40
        and three_day_macd_neg
        and row["Close"] < row["50EMA"]
    ):
        return "Bearish"

    # --- High Volatility ---
    elif row["ATR"] > row["ATR_50"] and abs(row["MACD"] - row["Signal"]) > 50:
        return "High Volatility"

    # --- Mean Reversion ---
    elif (
        row["Close"] < row["BB_Upper"] 
        and row["Close"] > row["BB_Lower"] 
        and 40 < row["RSI"] < 60
        and prev_count.get("Mean Reversion", 0) >= 3
    ):
        return "Mean Reversion"

    # --- Hold trend ---
    elif prev_regime in ["Bullish", "Bearish"] and prev_count.get(prev_regime, 0) < 5:
        return prev_regime

    else:
        return "Neutral"

# --- Step 4: Apply Enhanced Regime Classification ---
df.dropna(inplace=True)
df["Enhanced_Regime"] = None

prev_regime = None
prev_count = {"Bullish": 0, "Bearish": 0, "Mean Reversion": 0, "High Volatility": 0, "Neutral": 0}

for index, (idx, row) in enumerate(df.iterrows()):
    new_regime = detect_enhanced_regime(row, prev_regime, prev_count, df, index)
    
    if new_regime == prev_regime:
        prev_count[new_regime] += 1
    else:
        prev_count = {"Bullish": 0, "Bearish": 0, "Mean Reversion": 0, "High Volatility": 0, "Neutral": 0}

    df.at[idx, "Enhanced_Regime"] = new_regime
    prev_regime = new_regime

# --- Step 5: Visualization ---
plt.figure(figsize=(14, 8))
sns.scatterplot(data=df, x=df.index, y="Close", hue="Enhanced_Regime", palette="Set1", alpha=0.7, s=80)
plt.plot(df.index, df["50EMA"], label="50 EMA", color="orange", linewidth=2, linestyle='--')
plt.plot(df.index, df["200EMA"], label="200 EMA", color="purple", linewidth=2, linestyle='--')
plt.title("Refined Market Regime Detection (Earlier Bearish Signals)", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("NIFTY50 Close Price", fontsize=12)
plt.legend(title="Regimes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True)
plt.tight_layout()
# Save the chart before displaying
plt.savefig("feature_development/regime_detection/dev/NIFTY50_Enhanced_Regime_Chart.png", dpi=300)  # You can also use .pdf, .svg etc.

plt.show()


# --- Step 6: Save Processed Data ---
df.to_csv("feature_development/regime_detection/dev/NIFTY50_Refined_Bearish_Regime_Detection.csv")
