import os
import pandas as pd
import numpy as np
from get_data.yfinance_multistock_data import download_and_split,download_and_split_index
from overlays.indicators_v1 import (
    MovingAverageOverlay, RSIOverlay, VolumeOverlay, MACDOverlay,
    BollingerBandsOverlay, StochasticOscillatorOverlay, ATROverlay,
    EMAOverlay, FibonacciOverlay, VWAPOverlay
)
from overlays.zigzagsr_v1 import ZigzagSR
from overlays.regime_detection_v1 import EnhancedRegimeOverlay
from utilities.fib_utils import last_price_fib_info
from candelstick_base.candlestick_chart_v1 import CandlestickChart
from functools import reduce



class AIMLFeaturePipeline:
    def __init__(self, future_return=5,
                 data_dir="strategy_development/strategies/dev/data",
                 chart_dir="strategy_development/strategies/dev/chart"):
        self.data_dir = data_dir
        self.chart_dir = chart_dir
        self.future_return = future_return
        os.makedirs(self.data_dir, exist_ok=True)

    # -------------------------------
    #   Stock-Level Features Only
    # -------------------------------
    def generate_stock_features(self, tickers, start=None, end=None, period=None,add_index=False,show=False):
        """
        Strategy: Full Indicator Suite
        ------------------------------
        - Displays nearly all available overlays and subplots for comprehensive analysis
        - No filtering; for manual visual review
        - Includes trend, momentum, volume, volatility, regime detection, support/resistance, and Fibonacci zones
        """
        final_dfs = []
        if add_index:
            # Shift start 7 months back for index for the 200 EMA lookukback in the Enhanced regime
            start_index = (pd.to_datetime(start) - pd.DateOffset(months=9)).strftime("%Y-%m-%d")
            df_index = self.generate_index_features(start=start_index, end=end, period=period)
    
        dfs = download_and_split(tickers, start=start, end=end, period=period)

        for ticker, df in dfs.items():
            try:
                if df.empty or df.isna().all().all():
                    print(f"{ticker} returned empty or invalid data — skipped")
                    continue

                df.columns.name = None
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                df['Ticker'] = ticker
                df['future_return'] = df['Close'].shift(-self.future_return) / df['Close'] - 1
                df['Range_pct'] = (df['High'] - df['Low']) / df['Close']

                chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=show) # if you want to view plot

                # Trend & Regime
                chart.add_overlay(EMAOverlay(window=20, color="green", show=True))
                chart.add_overlay(EMAOverlay(window=50, color="red", show=True))
                chart.add_overlay(BollingerBandsOverlay(show=True))
                # chart.add_overlay(EnhancedRegimeOverlay(show=True))
                chart.add_overlay(FibonacciOverlay(lookback=50, show=True))

                # Support/Resistance
                chart.add_overlay(ZigzagSR(
                    min_peak_distance=8, min_peak_prominence=10,
                    zone_merge_tolerance=0.007, max_zones=8,
                    color_zone="blue", alpha_zone=0.15,
                    show=True, show_fibo=False, show_trendline=True,
                    show_only_latest_fibo=False
                ))

                # Volume & Volatility
                chart.add_subplot(VolumeOverlay(), height_ratio=1)
                chart.add_subplot(VWAPOverlay(show=True), height_ratio=1)
                chart.add_subplot(ATROverlay(), height_ratio=1)

                # Momentum
                chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
                chart.add_subplot(MACDOverlay(show=True), height_ratio=1)
                chart.add_subplot(StochasticOscillatorOverlay(show=True), height_ratio=1)

                # --- Compute all overlays/subplots ---
                df = chart.only_df()
                # Merge index with stock
                if add_index and df_index is not None and not df_index.empty:
                    # Ensure Date columns are datetime
                    df_index['Date_x'] = pd.to_datetime(df_index['Date_x'])
                    df = df.reset_index().rename(columns={'index': 'Date'})  # stock Date
                    df = pd.merge(df, df_index, left_on='Date', right_on='Date_x', how='left')

                df.set_index('Date', inplace=True)
                final_dfs.append(df)

                if show:
                    print('Show the stock on chart')
                    # --- Calculate indicator summaries ---
                    fib_info = last_price_fib_info(df)
                    fib_percent = round(fib_info.get("fib_percent", 0), 2)
                    fibo_status = df.get("Fibo_Status_Last_Close", pd.Series(["N/A"])).iloc[-1]
                    ema20 = df["EMA_20"].iloc[-1]
                    ema50 = df["EMA_50"].iloc[-1]
                    ema_trend = "Bullish" if ema20 > ema50 else "Bearish"

                    rsi = df["RSI_14"].iloc[-1]
                    rsi_signal = "Oversold" if rsi < 40 else "Overbought" if rsi > 60 else "Neutral"

                    vwap = df["VWAP"].iloc[-1]
                    close = df["Close"].iloc[-1]
                    vol = df["Volume"].iloc[-1]
                    avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
                    vol_signal = "High" if vol > avg_vol else "Low/Normal"

                    # --- Annotation text for quick summary ---
                    note_text = (
                        f"Fib%: {fib_percent}% ({fibo_status})\n"
                        f"EMA Trend: {ema_trend}\n"
                        f"RSI: {rsi:.1f} ({rsi_signal})\n"
                        f"VWAP: {'Above' if close > vwap else 'Below'}\n"
                        f"Volume: {vol_signal}"
                    )

                    chart.add_text(note_text, x=1, y=0.98, fontsize=11, color="black", bbox=True)
                    # df.to_csv(f"{self.data_dir}/{ticker}_full_suite.csv")
                    # --- Save final chart ---
                    save_path = f"{self.chart_dir}/{ticker}_full_suite.png"
                    chart.plot(save_path=save_path)
                    print(f"Saved full-suite chart for {ticker} — Fib%: {fib_percent}% at {save_path}")

            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                
        if final_dfs:
            df_all = pd.concat(final_dfs, axis=0)  # stack rows
            df_all.to_csv(f"{self.data_dir}/all_stocks_full_suite.csv", index=True)
            print(f"Saved concatenated full-suite for all tickers: {df_all.shape[0]} rows")


    def generate_index_features(self, start=None, end=None, period=None,show=False):
        """
        Generate key features for NIFTY and BANKNIFTY to provide market context.
        Features included:
            - EMA20, EMA50
            - RSI14
            - Trend label (Bullish/Bearish)
            - Market regime (Bullish/Bearish/Volatile)
        Saves CSV to data_dir as index_features.csv
        """
        tickers = ["^NSEI", "^NSEBANK"]  # NIFTY and BANKNIFTY
        dfs = download_and_split_index(tickers, start=start, end=end, period=period)
        
        index_features_list = []

        for ticker, df in dfs.items():
            if df.empty or df.isna().all().all():
                print(f"{ticker} returned empty or invalid data — skipped")
                continue

            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df = df.copy()

            chart = CandlestickChart(df, ticker=ticker, show_candles=False, show=show)

            # Add only essential overlays
            chart.add_overlay(EMAOverlay(window=20, show=True))
            chart.add_overlay(EMAOverlay(window=50, show=True))
            chart.add_overlay(EMAOverlay(window=200, show=True))
            chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
            chart.add_subplot(MACDOverlay(show=True), height_ratio=1)
            chart.add_subplot(ATROverlay(window=14), height_ratio=1)
            chart.add_overlay(BollingerBandsOverlay(show=True))
            chart.add_subplot(VolumeOverlay(), height_ratio=1)
            chart.add_subplot(VWAPOverlay(show=True), height_ratio=1)
            chart.add_subplot(StochasticOscillatorOverlay(show=True), height_ratio=1)
            chart.add_overlay(EnhancedRegimeOverlay(show=True)) # !!!!!!!!!!!!!!!!!!this cut's the data by 6 months
            chart.plot()

            df["Date"] = df.index  # explicitly add Date column for merging

            df_ind = chart.only_df()
            # Prefix all columns with ticker-friendly name
            prefix = "NIFTY" if ticker == "^NSEI" else "BANKNIFTY"
            # Dynamic renaming for all indicator columns
            df_ind = df_ind.rename(columns={col: f"{prefix}_{col}" for col in df_ind.columns if col != "Date"})

            # Add Trend column
            df_ind[f"{prefix}_Trend"] = np.where(
                df_ind[f"{prefix}_EMA_20"] > df_ind[f"{prefix}_EMA_50"], "Bullish", "Bearish"
            )

            # Append to list for merging later
            index_features_list.append(df_ind)
        # Merge NIFTY and BANKNIFTY on Date
        if index_features_list:
            # Merge on index (datetime) instead of 'Date' column
            df_index = reduce(lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True, how="outer"
            ), index_features_list)

            df_index.sort_index(inplace=True)
            # Optional: reset index to have a 'Date' column
            # df_index = df_index.reset_index()
            df_index.rename(columns={'Date': 'Date_x'}, inplace=True)
            # df_index.drop(['Date'],inplace=True, axis=1)
            # Save CSV
            # save_path = os.path.join(self.data_dir, "index_features.csv")
            # df_index.to_csv(save_path, index=False)
            # print(f"Index features saved to {save_path}")
            return df_index
        else:
            print("No index features generated. Check your downloaded data.")
            df_index = pd.DataFrame()
            return None

