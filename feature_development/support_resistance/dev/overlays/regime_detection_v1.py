import pandas as pd
import ta
import matplotlib.pyplot as plt
import seaborn as sns


class BaseOverlay:
    def plot(self, ax, df):
        raise NotImplementedError("Overlay must implement a plot method.")


class EnhancedRegimeOverlay(BaseOverlay):
    """
    Overlay that computes technical indicators (with unique names) and detects enhanced regimes.
    - compute_indicators(df) : computes indicators and returns df (indicator cols end with _for_RD)
    - compute(df)            : computes indicators + Enhanced_Regime_for_RD column (returns processed df)
    - plot(ax, df)           : overlays regime scatter + 50/200 EMA lines on provided axis
    """

    def __init__(self, show=True, palette="Set1", title="Refined Market Regime Detection (Earlier Bearish Signals)"):
        self.show = show
        self.palette = palette
        self.title = title

    # --- extra helper method requested ---
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the technical indicators and store them with unique names to avoid clashes.
        Returns a new DataFrame (copy) with indicator columns appended (suffix: _for_RD).
        """
        required = ["Close", "High", "Low"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Input df must contain column '{c}'")

        df = df.copy()

        # EMAs
        df["50_EMA_for_RD"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
        df["200_EMA_for_RD"] = ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator()

        # RSI
        df["RSI_for_RD"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

        # ATR and rolling ATR mean
        df["ATR_for_RD"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
        df["ATR_50_for_RD"] = df["ATR_for_RD"].rolling(50).mean()

        # MACD
        macd_obj = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD_for_RD"] = macd_obj.macd()
        df["Signal_for_RD"] = macd_obj.macd_signal()
        df["MACD_Hist_for_RD"] = df["MACD_for_RD"] - df["Signal_for_RD"]

        # Bollinger Bands
        bb_obj = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_Upper_for_RD"] = bb_obj.bollinger_hband()
        df["BB_Lower_for_RD"] = bb_obj.bollinger_lband()

        return df

    def detect_enhanced_regime(self, row, prev_regime, prev_count, df, index):
        """
        EXACT regime logic preserved — only column names reference the _for_RD fields.
        """
        # Last 3 days of MACD_Hist for momentum confirmation
        if index >= 3:
            macd_hist_3d = df["MACD_Hist_for_RD"].iloc[index - 2:index + 1].values
            three_day_macd_neg = all(macd < 0 for macd in macd_hist_3d)
        else:
            three_day_macd_neg = False

        # --- Bullish Regime ---
        if (
            row["50_EMA_for_RD"] > row["200_EMA_for_RD"]
            and row["MACD_for_RD"] > row["Signal_for_RD"]
            and row["RSI_for_RD"] > 55
        ):
            return "Bullish"

        # --- Bearish Regime (Earlier Detection) ---
        elif (
            row["50_EMA_for_RD"] < row["200_EMA_for_RD"]
            and row["MACD_for_RD"] < row["Signal_for_RD"]
            and row["RSI_for_RD"] < 40
            and three_day_macd_neg
            and row["Close"] < row["50_EMA_for_RD"]
        ):
            return "Bearish"

        # --- High Volatility ---
        elif row["ATR_for_RD"] > row["ATR_50_for_RD"] and abs(row["MACD_for_RD"] - row["Signal_for_RD"]) > 50:
            return "High Volatility"

        # --- Mean Reversion ---
        elif (
            row["Close"] < row["BB_Upper_for_RD"]
            and row["Close"] > row["BB_Lower_for_RD"]
            and 40 < row["RSI_for_RD"] < 60
            and prev_count.get("Mean Reversion", 0) >= 3
        ):
            return "Mean Reversion"

        # --- Hold trend ---
        elif prev_regime in ["Bullish", "Bearish"] and prev_count.get(prev_regime, 0) < 5:
            return prev_regime

        else:
            return "Neutral"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute indicators (unique names) and Enhanced_Regime_for_RD column.
        Returns processed DataFrame (a copy with NaNs dropped prior to regime detection).
        """
        df = self.compute_indicators(df)

        # drop rows with NaN in any of the computed indicator columns so detection has full data
        df = df.dropna().copy()

        # initialize
        df["Enhanced_Regime_for_RD"] = None
        prev_regime = None
        prev_count = {"Bullish": 0, "Bearish": 0, "Mean Reversion": 0, "High Volatility": 0, "Neutral": 0}

        # preserve original logic of counting / resetting
        for index, (idx, row) in enumerate(df.iterrows()):
            new_regime = self.detect_enhanced_regime(row, prev_regime, prev_count, df, index)

            if new_regime == prev_regime:
                prev_count[new_regime] += 1
            else:
                prev_count = {"Bullish": 0, "Bearish": 0, "Mean Reversion": 0, "High Volatility": 0, "Neutral": 0}

            df.at[idx, "Enhanced_Regime_for_RD"] = new_regime
            prev_regime = new_regime

        return df

    def plot(self, ax, df: pd.DataFrame):
        """
        Plot the regime overlay on the provided matplotlib axis.
        The method calls compute(df) internally, so pass the raw price df (with Close/High/Low).
        """
        df_proc = self.compute(df)

        if not self.show:
            return  # nothing to draw

        # scatter by regime
        sns.scatterplot(
            data=df_proc,
            x=df_proc.index,
            y="Close",
            hue="Enhanced_Regime_for_RD",
            palette=self.palette,
            alpha=0.7,
            s=80,
            ax=ax,
        )

        # EMAs (using the renamed columns)
        ax.plot(df_proc.index, df_proc["50_EMA_for_RD"], label="50 EMA (RD)", linewidth=2, linestyle="--")
        ax.plot(df_proc.index, df_proc["200_EMA_for_RD"], label="200 EMA (RD)", linewidth=2, linestyle="--")

        ax.set_title(self.title, fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.grid(True)
        # Place legend: combine regime hue legend (created by seaborn) and EMA lines
        ax.legend(title="Regimes / Lines", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
