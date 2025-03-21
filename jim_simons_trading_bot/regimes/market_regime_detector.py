import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta


class MarketRegimeDetector:
    def __init__(self, file_path, lookback=600):
        """
        Initializes the MarketRegimeDetector class.
        :param file_path: Path to the NIFTY50 data CSV file.
        :param lookback: Number of recent days to analyze.
        """
        self.file_path = file_path
        self.lookback = lookback
        self.df = None  # DataFrame to hold market data
        self.save_plot_path = "plots/NIFTY50PLOT.png"

    def load_data(self):
        """Loads the NIFTY50 data and selects the last 'lookback' days."""
        self.df = pd.read_csv(self.file_path, parse_dates=["Date"], index_col="Date")
        self.df = self.df.tail(self.lookback)

    def compute_technical_indicators(self):
        """Computes key technical indicators for market regime detection."""
        self.df["50EMA"] = ta.ema(self.df["Close"], length=50)
        self.df["200EMA"] = ta.ema(self.df["Close"], length=200)
        self.df["RSI"] = ta.rsi(self.df["Close"], length=14)
        self.df["ATR"] = ta.atr(self.df["High"], self.df["Low"], self.df["Close"], length=14)
        self.df["ATR_50"] = self.df["ATR"].rolling(50).mean()  # Rolling ATR mean for volatility
        macd = ta.macd(self.df["Close"], fast=12, slow=26, signal=9)
        self.df["MACD"] = macd["MACD_12_26_9"]
        self.df["Signal"] = macd["MACDs_12_26_9"]
        self.df["MACD_Hist"] = self.df["MACD"] - self.df["Signal"]  # MACD Histogram for momentum
        bbands = ta.bbands(self.df["Close"], length=20)
        self.df["BB_Upper"] = bbands["BBU_20_2.0"]
        self.df["BB_Lower"] = bbands["BBL_20_2.0"]
        self.df.dropna(inplace=True)

    def detect_enhanced_regime(self, row, prev_regime, prev_count, index):
        """
        Identifies the market regime for a given row in the DataFrame.
        Implements trend-following, early bearish detection, and mean reversion rules.
        """
        # Get the last 3 days of MACD_Hist for momentum confirmation
        if index >= 3:
            macd_hist_3d = self.df["MACD_Hist"].iloc[index-2:index+1].values
            three_day_macd_neg = all(macd < 0 for macd in macd_hist_3d)
        else:
            three_day_macd_neg = False  # Not enough data yet

        # --- Bullish Regime ---
        if row["50EMA"] > row["200EMA"] and row["MACD"] > row["Signal"] and row["RSI"] > 55:
            return "Bullish"

        # --- Bearish Regime (EARLIER DETECTION) ---
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

        # --- Mean Reversion (MUST HOLD INSIDE BOLLINGER BANDS FOR MULTIPLE DAYS) ---
        elif (
            row["Close"] < row["BB_Upper"] 
            and row["Close"] > row["BB_Lower"] 
            and 40 < row["RSI"] < 60
            and prev_count.get("Mean Reversion", 0) >= 3
        ):
            return "Mean Reversion"

        # --- HOLD TREND TO PREVENT FREQUENT SWITCHES ---
        elif prev_regime in ["Bullish", "Bearish"] and prev_count.get(prev_regime, 0) < 5:
            return prev_regime  # Maintain trend if it's not been long enough

        else:
            return "Neutral"  # Default if no strong signal

    def apply_regime_classification(self):
        """Applies market regime classification for all data points."""
        self.df["Enhanced_Regime"] = None  # Initialize regime column

        prev_regime = None
        prev_count = {"Bullish": 0, "Bearish": 0, "Mean Reversion": 0, "High Volatility": 0, "Neutral": 0}

        for index, (idx, row) in enumerate(self.df.iterrows()):
            new_regime = self.detect_enhanced_regime(row, prev_regime, prev_count, index)
            
            # Count consecutive days in the same regime
            if new_regime == prev_regime:
                prev_count[new_regime] += 1
            else:
                prev_count = {"Bullish": 0, "Bearish": 0, "Mean Reversion": 0, "High Volatility": 0, "Neutral": 0}

            self.df.at[idx, "Enhanced_Regime"] = new_regime
            prev_regime = new_regime
        

    def plot_market_regime(self):
        """Visualizes market regimes on a price chart."""
        plt.figure(figsize=(14, 8))

        # Scatterplot with regime-based coloring
        sns.scatterplot(data=self.df, x=self.df.index, y="Close", hue="Enhanced_Regime", palette="Set1", alpha=0.7, s=80)

        # Plot Moving Averages
        plt.plot(self.df.index, self.df["50EMA"], label="50 EMA", color="orange", linewidth=2, linestyle='--')
        plt.plot(self.df.index, self.df["200EMA"], label="200 EMA", color="purple", linewidth=2, linestyle='--')

        # Titles and labels
        plt.title("Refined Market Regime Detection (Earlier Bearish Signals)", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("NIFTY50 Close Price", fontsize=12)
        plt.legend(title="Regimes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True)

        # # Show Plot
        # plt.tight_layout()
        # plt.show()

        # Save Plot
        plt.savefig(self.save_plot_path, dpi=300, bbox_inches="tight")
    def save_processed_data(self, output_file="data/NIFTY50_Refined_Bearish_Regime_Detection.csv"):
        """Saves processed data with detected market regimes."""
        self.df.to_csv(output_file)
        print(f"Processed data saved to {output_file}")

    def run_regime_detector(self):
        """Runs the entire pipeline: load data, compute indicators, detect regimes, and visualize results."""
        print("Loading data...")
        self.load_data()

        print("Computing technical indicators...")
        self.compute_technical_indicators()

        print("Applying regime classification...")
        self.apply_regime_classification()

        print("Plotting market regime detection...")
        self.plot_market_regime()

        print("Saving processed data...")
        self.save_processed_data()

        return self.df


# === Usage Example ===
if __name__ == "__main__":
    detector = MarketRegimeDetector(file_path="data/NIFTY50_1d_5y.csv")
    detector.run_regime_detector()
