# main.py

import yfinance as yf
import pandas as pd
import os
from candelstick_base.candlestick_chart_v1 import CandlestickChart
from overlays.indicators_v1 import MovingAverageOverlay, RSIOverlay,VolumeOverlay,MACDOverlay,BollingerBandsOverlay,StochasticOscillatorOverlay,ATROverlay,EMAOverlay,FibonacciOverlay,VWAPOverlay
# from overlays.support_resistance import SupportResistanceZones # added in zigzagsr
from overlays.zigzagsr_v1 import ZigzagSR
from overlays.regime_detection_v1 import EnhancedRegimeOverlay
from get_data.yfinance_multistock_data import download_and_split
from utilities.fib_utils import last_price_fib_info

class ChartPipeline:
    def __init__(self, data_dir="strategy_development/fibo_strategies/dev/data",
                       chart_dir="strategy_development/fibo_strategies/dev/charts"):
        self.data_dir = data_dir
        self.chart_dir = chart_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.chart_dir, exist_ok=True)

    def plot(self, ticker, period=None, start=None, end=None, overlay=True):
        """
            Plot charts for given tickers and save them.
            This function is designed specifically for index plotting.
        """
        if period is None and (start is None or end is None):
            period = "1y"
        df = yf.download(ticker, period=period,start=start,end=end, interval="1d", auto_adjust=False, multi_level_index=False)
        chart = CandlestickChart(df, ticker=ticker, show_candles=True,show=True)
        chart.add_overlay(EnhancedRegimeOverlay(show=True))
        chart.plot()

    def strategy_1(self, tickers, fib_level_filter=[50, 65], start=None, end=None, period=None):  # 1y, 6mo
        """
        Only save charts and data for tickers whose last price is within fib_level_filter range.
        !!! Use at least 3-4 months of data
        """
        dfs = download_and_split(tickers, start=start, end=end, period=period)
        for ticker, df in dfs.items():
            if df.empty or df.isna().all().all():
                print(f"{ticker} returned empty or invalid data — skipped")
                continue

            # Standardize dataframe
            df.columns.name = None
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Create chart object
            chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=True)

            # Add indicators
            chart.add_overlay(EMAOverlay(window=50, color="red", show=True))
            chart.add_overlay(EMAOverlay(window=20, color="green", show=True))
            chart.add_overlay(FibonacciOverlay(lookback=30, show=True))
            zigzag = ZigzagSR(
                min_peak_distance=8, min_peak_prominence=10,
                zone_merge_tolerance=0.007, max_zones=8,
                color_zone="green", alpha_zone=0.1,
                show=True, show_fibo=False, show_trendline=False,
                show_only_latest_fibo=False
            )
            chart.add_overlay(zigzag)
            chart.add_subplot(VolumeOverlay(), height_ratio=1)

            # Compute overlays/subplots
            df = chart.only_df()
            fib_info = last_price_fib_info(df)
            fib_percent = round(fib_info["fib_percent"], 2)
            fibo_status = df["Fibo_Status_Last_Close"].iloc[-1]

            if fib_level_filter[0] <= fib_percent <= fib_level_filter[1]:
                print(f"{ticker}: last price is at {fib_percent:.2f}% Fib — saving chart & data")
                df_to_save = chart.only_df()
                # df_to_save.to_csv(f"{self.data_dir}/{ticker}_data.csv")

                # --- Add Fib% as text on chart ---
                note_text = f"Fib%: {fib_percent:.2f}%\nStatus: {fibo_status}"
                chart.add_text(
                    text=note_text,
                    x=0.01,      # top-left corner
                    y=0.99,
                    fontsize=12,
                    color="black",
                    bbox=True
                )

                # Plot chart (filename can just be ticker name)
                chart.plot(f'{self.chart_dir}/{ticker}.png')
            else:
                print(f"{ticker}: last price at {fib_percent:.2f}% Fib — skipped")


    def strategy_2(self, tickers, fib_level_filter=[50, 61], start=None, end=None, period=None):
        """ fibpercent_strategy_2:

            This method scans a list of tickers and saves charts & data for those whose 
            last close price is within a specified Fibonacci retracement range (default 50–61%). 
            It also analyzes key technical indicators to record signals for strategy observation.

            Indicators & Observations included:

            1. Fibonacci Level (Fib%):
            - Filters stocks where last price falls within the selected Fib range.
            - Indicates potential support/resistance zones.

            2. EMA Trend:
            - Uses 20 EMA and 50 EMA to determine short-term vs medium-term trend.
            - Bullish if EMA20 > EMA50, Bearish if EMA20 < EMA50.

            3. RSI (14-period):
            - Momentum indicator to identify overbought/oversold conditions.
            - RSI > 60 → Overbought, RSI < 40 → Oversold, 40–60 → Neutral.

            4. Volume Analysis:
            - Compares last volume with 20-period average.
            - High volume → strong confirmation of move; Low/normal → weak confirmation.

            5. ZigZag / Swing Points:
            - Highlights recent peaks and troughs to identify potential support/resistance clusters.

            6. Notes on Chart:
            - All the above signals are summarized and annotated directly on the chart
                for easy visual inspection, eliminating the need to check console output.

            7. Usage:
            - Recommended to use at least 3–4 months of historical data for meaningful analysis.
            - Saves chart images and CSVs for all filtered tickers.

            Example Output Annotation on Chart:

            Fib%: 53.2%
            EMA: Bullish
            RSI: Overbought (62)
            Volume: High (1.5M)"""

        dfs = download_and_split(tickers, start=start, end=end, period=period)
        
        for ticker, df in dfs.items():
            if df.empty or df.isna().all().all():
                print(f"{ticker} returned empty or invalid data — skipped")
                continue
            
            # Standardize df
            df.columns.name = None
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Create chart object
            chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=True)
            
            # Add indicators
            chart.add_overlay(EMAOverlay(window=50, color="red", show=True))
            chart.add_overlay(EMAOverlay(window=20, color="green", show=True))
            chart.add_overlay(FibonacciOverlay(lookback=30, show=True))
            chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
            chart.add_subplot(VolumeOverlay(), height_ratio=1)
            chart.add_overlay(ZigzagSR(min_peak_distance=8, min_peak_prominence=10,
                                    zone_merge_tolerance=0.007, max_zones=8,
                                    color_zone="green", alpha_zone=0.1,
                                    show=True, show_fibo=False, show_trendline=False,
                                    show_only_latest_fibo=False))
            
            # Extract final dataframe
            df = chart.only_df()
            
            # --- Fibonacci Filtering ---
            fib_info = last_price_fib_info(df)
            fib_percent = round(fib_info["fib_percent"], 2)
            fibo_status = df["Fibo_Status_Last_Close"].iloc[-1]
            
            if fib_level_filter[0] <= fib_percent <= fib_level_filter[1]:
                print(f"{ticker}: last price at {fib_percent}% Fib — saving chart & data")
                
                # # Save CSV for strategy analysis
                # df_to_save = chart.only_df()
                # df_to_save.to_csv(f"{self.data_dir}/{ticker}_data.csv")
                
                # --- Strategy Observations ---
                ema20 = df["EMA_20"].iloc[-1]
                ema50 = df["EMA_50"].iloc[-1]
                ema_trend = "Bullish" if ema20 > ema50 else "Bearish"

                rsi = df["RSI_14"].iloc[-1]
                rsi_signal = "Oversold" if rsi < 40 else "Overbought" if rsi > 60 else "Neutral"

                avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
                last_vol = df["Volume"].iloc[-1]
                vol_signal = "High" if last_vol > avg_vol else "Low/Normal"

                # Create a note string
                note_text = (
                    f"Fib%: {fib_percent}%\n"
                    f"Fibo Status: {fibo_status}\n"
                    f"EMA: {ema_trend}\n"
                    f"RSI: {rsi_signal} ({rsi:.1f})\n"
                    f"Volume: {vol_signal} ({last_vol:.0f})"
                )

                # Add annotation to the chart (top-left corner)
                chart.add_text(note_text)
                chart.plot(save_path=f'{self.chart_dir}/{ticker}_{fib_percent}.png')

            else:
                print(f"{ticker}: last price at {fib_percent}% Fib — skipped")


    def strategy_3(self, tickers, start=None, end=None, period=None):
        dfs = download_and_split(tickers, start=start, end=end, period=period)

        for ticker, df in dfs.items():
            if df.empty or df.isna().all().all():
                print(f"{ticker} returned empty or invalid data — skipped")
                continue

            df.columns.name = None
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Create chart and add overlays (this also calculates indicator columns)
            chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=True)
            chart.add_overlay(EMAOverlay(window=20, color="green", show=True))
            chart.add_overlay(EMAOverlay(window=50, color="red", show=True))
            chart.add_overlay(FibonacciOverlay(lookback=30, show=True))
            chart.add_overlay(VWAPOverlay(show=True))
            chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
            chart.add_subplot(VolumeOverlay(), height_ratio=1)
            chart.add_overlay(ZigzagSR(
                min_peak_distance=8, min_peak_prominence=10,
                zone_merge_tolerance=0.007, max_zones=8,
                color_zone="green", alpha_zone=0.1,
                show=True, show_fibo=False, show_trendline=True,
                show_only_latest_fibo=False))

            # Get DataFrame with indicators
            df = chart.only_df()

            # Access indicators now
            try:
                fib_info = last_price_fib_info(df)
                fib_percent = round(fib_info["fib_percent"], 2)
                fibo_status = df.get("Fibo_Status_Last_Close", pd.Series(["N/A"])).iloc[-1]

                ema20 = df["EMA_20"].iloc[-1]
                ema50 = df["EMA_50"].iloc[-1]
                rsi = df["RSI_14"].iloc[-1]
                vwap = df["VWAP"].iloc[-1]
                close = df["Close"].iloc[-1]
                vol = df["Volume"].iloc[-1]
                avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
            except Exception as e:
                print(f"{ticker}: Failed to parse indicators — {e}")
                continue

            # Filter based on strategy conditions
            if ema20 > ema50 and 40 < rsi < 60 and close > vwap and vol > avg_vol:
                print(f"{ticker}: High-Quality Setup Found")

                # Add overlays for plotting with show=True now
                chart.add_text(
                    f"Fib%: {fib_percent}%\nEMA: Bullish\nRSI: {rsi:.1f} (Neutral)\nVWAP: Reclaimed\nVolume: High ({vol:.0f})"
                )
                chart.plot(save_path=f"{self.chart_dir}/{ticker}_strategy3.png")
            else:
                print(f"{ticker}: Does not meet high-quality criteria — skipped")



if __name__ == "__main__":
    # Create a singleton instance for convenience
    pipeline = ChartPipeline()
    # Plot NIFTY and BANKNIFTY
    pipeline.plot(["^NSEI"])
    pipeline.plot(["^NSEBANK"])

    # Filter tickers by Fib percent and save only if in 50-65%
    tickers = ["RELIANCE.NS", "TCS.NS"]
    fib_level_filter = [50, 65]
    pipeline.fibpercent(tickers, fib_level_filter=fib_level_filter)
