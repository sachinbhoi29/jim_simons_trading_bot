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

class genericStrategyPipeline:
    def __init__(self, data_dir="strategy_development/strategies/dev/data",
                       chart_dir="strategy_development/strategies/dev/charts"):
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


    def strategy_bullish_trend(self, tickers, start=None, end=None, period=None):
        """
        Bullish Regime Strategy:
        Focus on trend continuation using EMA, MACD, VWAP, RSI, and Fibonacci confluence.
        Ideal for NIFTY bullish or moderate uptrend phases.
        """
        dfs = download_and_split(tickers, start=start, end=end, period=period)

        for ticker, df in dfs.items():
            if df.empty or df.isna().all().all():
                print(f"{ticker}: Skipped — no data.")
                continue

            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=True)
            chart.add_overlay(EMAOverlay(window=20, color="green", show=True))
            chart.add_overlay(EMAOverlay(window=50, color="red", show=True))
            chart.add_subplot(MACDOverlay(show=True))
            chart.add_overlay(VWAPOverlay(show=True))
            chart.add_overlay(FibonacciOverlay(lookback=30, show=True))
            chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
            chart.add_subplot(VolumeOverlay(), height_ratio=1)
            chart.add_overlay(ZigzagSR(show=True))

            df = chart.only_df()

            try:
                ema20, ema50 = df["EMA_20"].iloc[-1], df["EMA_50"].iloc[-1]
                macd, macd_signal = df["MACD"].iloc[-1], df["MACD_signal"].iloc[-1]
                rsi = df["RSI_14"].iloc[-1]
                close, vwap = df["Close"].iloc[-1], df["VWAP"].iloc[-1]
                vol, avg_vol = df["Volume"].iloc[-1], df["Volume"].rolling(20).mean().iloc[-1]
                fib = last_price_fib_info(df)["fib_percent"]

                if (ema20 > ema50 and macd > macd_signal and
                    45 < rsi < 60 and close > vwap and vol > avg_vol and
                    45 <= fib <= 61):
                    print(f"{ticker}: ✅ Bullish continuation setup found (Fib {fib:.1f}%)")
                    chart.add_text(f"Bullish Setup\nFib {fib:.1f}% | EMA Bullish | RSI {rsi:.1f}")
                    chart.plot(save_path=f"{self.chart_dir}/{ticker}_bullish.png")
                else:
                    print(f"{ticker}: ❌ No bullish setup")
            except Exception as e:
                print(f"{ticker}: Indicator parse error — {e}")


    def strategy_bearish_trend(self, tickers, start=None, end=None, period=None):
        """
        Bearish Regime Strategy:
        Looks for short setups in a confirmed downtrend using EMA, MACD, RSI, VWAP, and Fib retracement.
        """
        dfs = download_and_split(tickers, start=start, end=end, period=period)
        for ticker, df in dfs.items():
            if df.empty or df.isna().all().all():
                continue

            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=True)
            chart.add_overlay(EMAOverlay(window=20, color="red", show=True))
            chart.add_overlay(EMAOverlay(window=50, color="blue", show=True))
            chart.add_subplot(MACDOverlay(show=True))
            chart.add_overlay(FibonacciOverlay(lookback=30, show=True))
            chart.add_overlay(VWAPOverlay(show=True))
            chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
            chart.add_subplot(VolumeOverlay(), height_ratio=1)
            chart.add_overlay(ZigzagSR(show=True))

            df = chart.only_df()

            try:
                ema20, ema50 = df["EMA_20"].iloc[-1], df["EMA_50"].iloc[-1]
                macd, macd_signal = df["MACD"].iloc[-1], df["MACD_signal"].iloc[-1]
                rsi = df["RSI_14"].iloc[-1]
                close, vwap = df["Close"].iloc[-1], df["VWAP"].iloc[-1]
                vol, avg_vol = df["Volume"].iloc[-1], df["Volume"].rolling(20).mean().iloc[-1]
                fib = last_price_fib_info(df)["fib_percent"]

                if (ema20 < ema50 and macd < macd_signal and
                    40 < rsi < 55 and close < vwap and vol > avg_vol and
                    38 <= fib <= 50):
                    print(f"{ticker}: ⚠️ Bearish continuation setup found (Fib {fib:.1f}%)")
                    chart.add_text(f"Bearish Setup\nFib {fib:.1f}% | EMA Bearish | RSI {rsi:.1f}")
                    chart.plot(save_path=f"{self.chart_dir}/{ticker}_bearish.png")
                else:
                    print(f"{ticker}: ❌ No bearish setup")
            except Exception as e:
                print(f"{ticker}: Indicator parse error — {e}")

    def strategy_neutral_breakout(self, tickers, start=None, end=None, period=None,
                                bb_window=20, bb_num_std=2, volume_multiple=1.5, tolerance=0.02):
        """
        Neutral Regime Strategy:
        Detects breakouts from low-volatility range using Bollinger Bands + Volume Burst.
        
        Parameters:
            tickers: list of tickers
            bb_window: Bollinger Bands window
            bb_num_std: Number of standard deviations
            volume_multiple: Multiplier for volume spike
            tolerance: % distance from BB considered as soft breakout
        """
        dfs = download_and_split(tickers, start=start, end=end, period=period)

        for ticker, df in dfs.items():
            if df.empty or df.isna().all().all():
                print(f"{ticker}: Skipped — no valid data")
                continue

            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Initialize chart and add overlays
            chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=True)
            chart.add_overlay(BollingerBandsOverlay(window=bb_window, num_std=bb_num_std, show=True))
            chart.add_subplot(VolumeOverlay(), height_ratio=1)
            chart.add_overlay(EMAOverlay(window=20, color="green", show=True))
            chart.add_overlay(EMAOverlay(window=50, color="red", show=True))

            df = chart.only_df()

            # Dynamic column names
            bb_upper_col = f"BB_upper_{bb_window}"
            bb_lower_col = f"BB_lower_{bb_window}"

            close = df["Close"].iloc[-1]
            vol = df["Volume"].iloc[-1]
            avg_vol = df["Volume"].rolling(20).mean().iloc[-1]

            # Debug print to see why tickers fail
            print(f"{ticker} | Close: {close:.2f} | BB Upper: {df[bb_upper_col].iloc[-1]:.2f} | BB Lower: {df[bb_lower_col].iloc[-1]:.2f} | Vol: {vol} | AvgVol: {avg_vol:.0f}")

            # Soft breakout: within tolerance distance of band
            soft_breakout = (
                (close >= df[bb_upper_col].iloc[-1] * (1 - tolerance)) or
                (close <= df[bb_lower_col].iloc[-1] * (1 + tolerance))
            )

            # Breakout condition: soft breakout + moderate volume spike
            breakout_condition = soft_breakout and vol >= avg_vol * volume_multiple

            if breakout_condition:
                direction = "Bullish" if close > df[bb_upper_col].iloc[-1] else "Bearish"
                print(f"{ticker}: 💥 {direction} breakout detected from range with volume spike")
                chart.add_text(f"{direction} Range Breakout + Volume Spike")
                chart.plot(save_path=f"{self.chart_dir}/{ticker}_neutral_breakout.png")
            else:
                print(f"{ticker}: ⚪ No breakout in neutral regime")



    def strategy_high_volatility(self, tickers, start=None, end=None, period=None):
        """
        High Volatility Regime Strategy:
        Fade extremes using ATR, RSI, and Fibonacci overshoot.
        """
        dfs = download_and_split(tickers, start=start, end=end, period=period)
        for ticker, df in dfs.items():
            if df.empty:
                continue

            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=True)
            chart.add_overlay(ATROverlay(window=14, show=True))
            chart.add_overlay(RSIOverlay(period=14, show=True))
            chart.add_overlay(FibonacciOverlay(lookback=30, show=True))
            chart.add_overlay(VWAPOverlay(show=True))
            chart.add_overlay(ZigzagSR(show=True))

            df = chart.only_df()

            atr = df["ATR_14"].iloc[-1]
            atr_avg = df["ATR_14"].rolling(20).mean().iloc[-1]
            rsi = df["RSI_14"].iloc[-1]
            fib = last_price_fib_info(df)["fib_percent"]
            close = df["Close"].iloc[-1]

            if (atr > atr_avg * 1.5) and ((rsi < 30 and fib >= 61) or (rsi > 70 and fib <= 38)):
                print(f"{ticker}: ⚡ Mean reversion opportunity detected (RSI {rsi:.1f}, Fib {fib:.1f}%)")
                chart.add_text(f"High Volatility Fade\nRSI {rsi:.1f} | Fib {fib:.1f}%")
                chart.plot(save_path=f"{self.chart_dir}/{ticker}_highvol.png")
            else:
                print(f"{ticker}: 🟣 No volatility fade setup")


    def trend_fibo_conf_strategy(self, tickers, start=None, end=None, period=None):
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


    def multi_confluence_strategy(self, tickers, start=None, end=None, period=None):
        """
        Multi-Confluence High-Quality Setup (MCHQS)
        Filters tickers based on trend, momentum, volume, and support/resistance confluence.
        """
        dfs = download_and_split(tickers, start=start, end=end, period=period)

        for ticker, df in dfs.items():
            if df.empty or df.isna().all().all():
                print(f"{ticker} returned empty or invalid data — skipped")
                continue

            df.columns.name = None
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Initialize chart and add multiple overlays
            chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=True)
            chart.add_overlay(EMAOverlay(window=20, color="green", show=True))
            chart.add_overlay(EMAOverlay(window=50, color="red", show=True))
            chart.add_subplot(MACDOverlay(show=True))
            chart.add_overlay(FibonacciOverlay(lookback=30, show=True))
            chart.add_overlay(VWAPOverlay(show=True))
            chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
            chart.add_subplot(StochasticOscillatorOverlay(), height_ratio=1)
            chart.add_subplot(VolumeOverlay(), height_ratio=1)
            chart.add_overlay(ZigzagSR(
                min_peak_distance=8, min_peak_prominence=10,
                zone_merge_tolerance=0.007, max_zones=8,
                color_zone="green", alpha_zone=0.1,
                show=True, show_fibo=False, show_trendline=True,
                show_only_latest_fibo=False))

            # Compute indicators
            df = chart.only_df()

            try:
                # Trend
                ema20 = df["EMA_20"].iloc[-1]
                ema50 = df["EMA_50"].iloc[-1]
                macd = df["MACD"].iloc[-1]
                macd_signal = df["MACD_signal"].iloc[-1]

                # Momentum
                rsi = df["RSI_14"].iloc[-1]
                stochastic_k = df["%K"].iloc[-1]
                stochastic_d = df["%D"].iloc[-1]

                # Volume and price
                close = df["Close"].iloc[-1]
                vwap = df["VWAP"].iloc[-1]
                vol = df["Volume"].iloc[-1]
                avg_vol = df["Volume"].rolling(20).mean().iloc[-1]

                # Fibonacci info
                fib_info = last_price_fib_info(df)
                fib_percent = round(fib_info["fib_percent"], 2)
                fibo_status = df.get("Fibo_Status_Last_Close", pd.Series(["N/A"])).iloc[-1]

                # ZigzagSR zones (latest support)
                zone_cols = [c for c in df.columns if "Zone" in c and "_low" in c]
                if zone_cols:
                    latest_zone_low = df[zone_cols[-1]].iloc[-1]
                    latest_zone_high = df[zone_cols[-1].replace("_low", "_high")].iloc[-1]
                else:
                    latest_zone_low = latest_zone_high = None

            except Exception as e:
                print(f"{ticker}: Failed to parse indicators — {e}")
                continue

            # --- High-Quality Long Setup Filter ---
            long_condition = (
                ema20 > ema50 and                   # Trend bullish
                macd > macd_signal and              # MACD confirms trend
                40 <= rsi <= 60 and                 # Neutral momentum
                stochastic_k > stochastic_d and     # Momentum confirmation
                close > vwap and                     # Price above VWAP
                vol > avg_vol and                    # Strong volume
                (latest_zone_low is None or close >= latest_zone_low)  # Near support
            )

            if long_condition:
                print(f"{ticker}: MCHQS High-Quality Long Setup Found")
                chart.add_text(
                    f"MCHQS Setup\nFib%: {fib_percent}%\nEMA: Bullish\nRSI: {rsi:.1f}\nVWAP Reclaimed\nVolume: High ({vol:.0f})"
                )
                chart.plot(save_path=f"{self.chart_dir}/{ticker}_MCHQS.png")
            else:
                print(f"{ticker}: Does not meet MCHQS criteria — skipped")


    def strategy_volume_burst(self, tickers, start=None, end=None, period=None,
                            volume_window=20, volume_multiple=4):
        """
        Volume Burst Strategy
        Detects extreme volume days with optional high-quality confluence.
        """
        dfs = download_and_split(tickers, start=start, end=end, period=period)

        for ticker, df in dfs.items():
            if df.empty or df.isna().all().all():
                print(f"{ticker} returned empty or invalid data — skipped")
                continue

            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            avg_vol = df["Volume"].rolling(volume_window).mean()
            latest_vol = df["Volume"].iloc[-1]

            if latest_vol >= avg_vol.iloc[-1] * volume_multiple:
                print(f"{ticker}: Volume Burst Detected! Volume = {latest_vol}, Avg = {avg_vol.iloc[-1]:.0f}")
                
                # Optional: Add chart with highlighted bar
                chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=True)
                chart.add_subplot(VolumeOverlay())
                chart.add_overlay(EMAOverlay(window=20, color="green", show=True))
                chart.add_overlay(EMAOverlay(window=50, color="red", show=True))
                chart.add_subplot(MACDOverlay(show=True))
                chart.add_overlay(FibonacciOverlay(lookback=30, show=True))
                chart.add_overlay(VWAPOverlay(show=True))
                chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
                chart.add_subplot(StochasticOscillatorOverlay(), height_ratio=1)
                chart.add_overlay(ZigzagSR(
                    min_peak_distance=8, min_peak_prominence=10,
                    zone_merge_tolerance=0.007, max_zones=8,
                    color_zone="green", alpha_zone=0.1,
                    show=True, show_fibo=False, show_trendline=True,
                    show_only_latest_fibo=False))
                chart.add_text(f"Volume Burst: {latest_vol / avg_vol.iloc[-1]:.1f}x avg")
                chart.plot(save_path=f"{self.chart_dir}/{ticker}_VolumeBurst.png")
            else:
                print(f"{ticker}: No significant volume spike — skipped")


if __name__ == "__main__":
    # Create a singleton instance for convenience
    pipeline = genericStrategyPipeline()
    # Plot NIFTY and BANKNIFTY
    pipeline.plot(["^NSEI"])
    pipeline.plot(["^NSEBANK"])

    # Filter tickers by Fib percent and save only if in 50-65%
    tickers = ["RELIANCE.NS", "TCS.NS"]
    fib_level_filter = [50, 65]
    pipeline.fibpercent(tickers, fib_level_filter=fib_level_filter)
