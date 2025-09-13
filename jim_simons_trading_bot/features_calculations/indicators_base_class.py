import pandas as pd
import numpy as np
import pandas_ta as ta



class IndicatorBase:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes with a market DataFrame (must include 'Close', 'High', 'Low', 'Volume').
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        self.df = df.copy()

    def compute_indicators(self):
        """Computes common technical indicators and appends them to the DataFrame."""
        self.df["50EMA"] = ta.ema(self.df["Close"], length=50)
        self.df["200EMA"] = ta.ema(self.df["Close"], length=200)
        self.df["RSI"] = ta.rsi(self.df["Close"], length=14)
        self.df["ATR"] = ta.atr(self.df["High"], self.df["Low"], self.df["Close"], length=14)
        self.df["ATR_50"] = self.df["ATR"].rolling(50).mean()

        macd = ta.macd(self.df["Close"], fast=12, slow=26, signal=9)
        self.df["MACD"] = macd["MACD_12_26_9"]
        self.df["Signal"] = macd["MACDs_12_26_9"]
        self.df["MACD_Hist"] = self.df["MACD"] - self.df["Signal"]

        bb = ta.bbands(self.df["Close"], length=20)
        self.df["BB_Upper"] = bb["BBU_20_2.0"]
        self.df["BB_Lower"] = bb["BBL_20_2.0"]
        self.df["BB_Width"] = (self.df["High"].rolling(20).max() - self.df["Low"].rolling(20).min()) / self.df["Close"].rolling(20).mean()

        self.df["OBV"] = (np.sign(self.df["Close"].diff()) * self.df["Volume"]).cumsum()
        self.df["Support"] = self.df["Low"].rolling(20).min()
        self.df["Resistance"] = self.df["High"].rolling(20).max()

        # Donchian Channel
        donchian_period = 20  # you can customize this
        self.df["Donchian_Upper"] = self.df["High"].rolling(window=donchian_period).max()
        self.df["Donchian_Lower"] = self.df["Low"].rolling(window=donchian_period).min()
        self.df["Donchian_Mid"] = (self.df["Donchian_Upper"] + self.df["Donchian_Lower"]) / 2

        # self.df.dropna(inplace=True)
        return self.df