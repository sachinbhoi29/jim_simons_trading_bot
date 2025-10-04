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
    def __init__(self, data_dir="strategy_development/fibo_50_65/dev/data",
                       chart_dir="strategy_development/fibo_50_65/dev/charts"):
        self.data_dir = data_dir
        self.chart_dir = chart_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.chart_dir, exist_ok=True)

    def plot(self, ticker, period="1y", overlay=True):
        """
        Plot charts for given tickers and save them.
        """
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, multi_level_index=False)
        chart = CandlestickChart(df, ticker=ticker, show_candles=True,show=True)
        chart.add_overlay(EnhancedRegimeOverlay(show=True))
        chart.plot()

    def fibpercent(self, tickers, fib_level_filter=[50, 65], period="6mo"): # 1y, 6mo
        """
        Only save charts and data for tickers whose last price is within fib_level_filter range.
        """
        dfs = download_and_split(tickers, period=period)
        for ticker, df in dfs.items():
            if df.empty or df.isna().all().all():
                print(f"{ticker} returned empty or invalid data — skipped")
                continue
            # Remove column name if it exists
            df.columns.name = None
            # Ensure Date is index
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            # print(df.head())
            chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=True)
            chart.add_overlay(EMAOverlay(window=50, color="red", show=True))
            chart.add_overlay(EMAOverlay(window=20, color="green", show=True))
            chart.add_overlay(FibonacciOverlay(lookback=50, show=True))
            zigzag = ZigzagSR(min_peak_distance=8, min_peak_prominence=10,
                                zone_merge_tolerance=0.007, max_zones=8,
                                color_zone="green", alpha_zone=0.1,
                                show=True, show_fibo=False, show_trendline=False,
                                show_only_latest_fibo=False)
            chart.add_overlay(zigzag)
            chart.add_subplot(VolumeOverlay(), height_ratio=1)
            df = chart.only_df()
            fib_info = last_price_fib_info(df)
            fib_percent = fib_info["fib_percent"]
            fib_percent = round(fib_percent, 2)
            if fib_level_filter[0] <= fib_percent <= fib_level_filter[1]:
                print(f"{ticker}: last price is at {fib_percent:.2f}% Fib — saving chart & data")
                df_to_save = chart.only_df()
                df_to_save.to_csv(f"{self.data_dir}/{ticker}_data.csv")
                chart.plot(f'strategy_development/fibo_50_65/dev/charts/{ticker}_{fib_percent}.png')
            else:
                print(f"{ticker}: last price at {fib_percent:.2f}% Fib — skipped")

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
