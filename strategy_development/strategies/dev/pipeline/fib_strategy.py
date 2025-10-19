# main.py

import yfinance as yf
import pandas as pd
import os
from candelstick_base.candlestick_chart_v1 import CandlestickChart
from overlays.indicators_v1 import MovingAverageOverlay, RSIOverlay,VolumeOverlay,MACDOverlay,BollingerBandsOverlay,StochasticOscillatorOverlay,ATROverlay,EMAOverlay,FibonacciOverlay,VWAPOverlay,FibonacciOverlayImproved
# from overlays.support_resistance import SupportResistanceZones # added in zigzagsr
from overlays.zigzagsr_v1 import ZigzagSR
from overlays.regime_detection_v1 import EnhancedRegimeOverlay
from get_data.yfinance_multistock_data import download_and_split
from utilities.fib_utils import last_price_fib_info

class fibPipeline:
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
                    x=1.00,      
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
            # chart.add_overlay(FibonacciOverlay(lookback=35, show=True))
            chart.add_overlay(FibonacciOverlayImproved(show=True))
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
                color_zone="green", alpha_zone=0.5,
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

    def strategy_no_filter(self, tickers, start=None, end=None, period=None):
        """
        Screening Strategy:
        -------------------
        - No strict filters, just compute and show all indicators
        - Annotates Fib%, EMA trend, RSI, Volume, VWAP, etc.
        - You manually review the charts and decide.
        """
        dfs = download_and_split(tickers, start=start, end=end, period=period)

        for ticker, df in dfs.items():
            if df.empty or df.isna().all().all():
                print(f"{ticker} returned empty or invalid data — skipped")
                continue

            df.columns.name = None
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=True)

            # === Add commonly used overlays and indicators ===
            chart.add_overlay(EMAOverlay(window=20, color="green", show=True))
            chart.add_overlay(EMAOverlay(window=50, color="red", show=True))
            chart.add_overlay(VWAPOverlay(show=True))
            chart.add_overlay(FibonacciOverlay(lookback=30, show=True))
            chart.add_overlay(ZigzagSR(
                min_peak_distance=8, min_peak_prominence=10,
                zone_merge_tolerance=0.007, max_zones=8,
                color_zone="blue", alpha_zone=0.15,
                show=True, show_fibo=False, show_trendline=True,
                show_only_latest_fibo=False
            ))
            chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
            chart.add_subplot(VolumeOverlay(), height_ratio=1)

            # Compute all overlays/subplots
            df = chart.only_df()

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

            # --- Annotation text for easy reading ---
            note_text = (
                f"Fib%: {fib_percent}% ({fibo_status})\n"
                f"EMA Trend: {ema_trend}\n"
                f"RSI: {rsi:.1f} ({rsi_signal})\n"
                f"VWAP: {'Above' if close > vwap else 'Below'}\n"
                f"Volume: {vol_signal}"
            )

            chart.add_text(note_text, x=0.01, y=0.98, fontsize=11, color="black", bbox=True)

            save_path = f"{self.chart_dir}/{ticker}_screen.png"
            chart.plot(save_path=save_path)
            print(f"Saved chart for {ticker} — Fib%: {fib_percent}%")

    def strategy_full_indicator_suite(self, tickers, start=None, end=None, period=None):
        """
        Strategy: Full Indicator Suite
        ------------------------------
        - Displays nearly all available overlays and subplots for comprehensive analysis
        - No filtering; for manual visual review
        - Includes trend, momentum, volume, volatility, regime detection, support/resistance, and Fibonacci zones
        """

        dfs = download_and_split(tickers, start=start, end=end, period=period)

        for ticker, df in dfs.items():
            if df.empty or df.isna().all().all():
                print(f"{ticker} returned empty or invalid data — skipped")
                continue

            df.columns.name = None
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=True)

            # === Trend & Regime Overlays ===
            chart.add_overlay(MovingAverageOverlay(window=20, color="blue", show=True))
            chart.add_overlay(MovingAverageOverlay(window=50, color="purple", show=True))
            chart.add_overlay(EMAOverlay(window=20, color="green", show=True))
            chart.add_overlay(EMAOverlay(window=50, color="red", show=True))
            chart.add_overlay(BollingerBandsOverlay(show=True))
            chart.add_overlay(EnhancedRegimeOverlay(show=True))

            # === Support/Resistance + Fibonacci ===
            chart.add_overlay(FibonacciOverlay(lookback=50, show=True))
            chart.add_overlay(ZigzagSR(
                min_peak_distance=8, min_peak_prominence=10,
                zone_merge_tolerance=0.007, max_zones=8,
                color_zone="blue", alpha_zone=0.15,
                show=True, show_fibo=True, show_trendline=True,
                show_only_latest_fibo=False
            ))

            # === Volume, VWAP, ATR ===
            chart.add_subplot(VolumeOverlay(), height_ratio=1)
            chart.add_subplot(VWAPOverlay(show=True), height_ratio=1)
            chart.add_subplot(ATROverlay(), height_ratio=1)

            # === Momentum Indicators ===
            chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
            chart.add_subplot(MACDOverlay(show=True), height_ratio=1)
            chart.add_subplot(StochasticOscillatorOverlay(show=True), height_ratio=1)

            # --- Compute all overlays/subplots ---
            df = chart.only_df()

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

            chart.add_text(note_text, x=0.01, y=0.98, fontsize=11, color="black", bbox=True)

            # --- Save final chart ---
            save_path = f"{self.chart_dir}/{ticker}_full_suite.png"
            chart.plot(save_path=save_path)
            print(f"Saved full-suite chart for {ticker} — Fib%: {fib_percent}%")



# if __name__ == "__main__":
#     # Create a singleton instance for convenience
#     pipeline = ChartPipeline()
#     # Plot NIFTY and BANKNIFTY
#     pipeline.plot(["^NSEI"])
#     pipeline.plot(["^NSEBANK"])

#     # Filter tickers by Fib percent and save only if in 50-65%
#     tickers = ["RELIANCE.NS", "TCS.NS"]
#     fib_level_filter = [50, 65]
#     pipeline.fibpercent(tickers, fib_level_filter=fib_level_filter)




if __name__ == '__main__':
    # Create a singleton instance for convenience
    fib_pipeline = fibPipeline()
    gen_pipeline = genericStrategyPipeline()
    # Plot NIFTY and BANKNIFTY
    # fib_pipeline.plot("^NSEI",start="2023-02-01",end="2024-05-02")#period='1y')#,start='2024-01-01',end='2025-01-01')
    # pipeline.plot(["^NSEBANK"])

    LARGE_CAP_TICKERS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
        "SBIN.NS", "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "KOTAKBANK.NS",
        "ITC.NS", "BHARTIARTL.NS", "MARUTI.NS", "SUNPHARMA.NS", "WIPRO.NS", "HCLTECH.NS",
        "M&M.NS", "NTPC.NS", "POWERGRID.NS", "TATAMOTORS.NS", "ULTRACEMCO.NS",
        "ADANIPORTS.NS", "CIPLA.NS", "DRREDDY.NS", "NESTLEIND.NS", "BAJAJFINSV.NS",
        "DIVISLAB.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "COALINDIA.NS", "GRASIM.NS",
        "HDFCLIFE.NS", "TECHM.NS", "UPL.NS", "BRITANNIA.NS", "EICHERMOT.NS", "HINDALCO.NS",
        "ONGC.NS", "BPCL.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS", "SBILIFE.NS", "ADANIENT.NS",
        "BAJAJ-AUTO.NS", "INDUSINDBK.NS", "ICICILOMBARD.NS", "TATACONSUM.NS",
        "ASIANPAINT.NS", "SHREECEM.NS", "DABUR.NS", "PIDILITIND.NS", "GODREJCP.NS",
        "HAVELLS.NS", "TORNTPHARM.NS", "COLPAL.NS", "BERGEPAINT.NS",
        "DLF.NS", "ZEEL.NS", "CHOLAFIN.NS", "MPHASIS.NS", "MINDTREE.NS",
        "GAIL.NS", "IOC.NS", "PETRONET.NS", "REC.NS", "PNBHOUSING.NS",
        "BANKBARODA.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "GLAND.NS", "ALKEM.NS",
        "MAXHEALTH.NS", "ICICIPRULI.NS", "ABB.NS", "SIEMENS.NS", "CUMMINSIND.NS",
        "BHEL.NS", "ASHOKLEY.NS", "CONCOR.NS", "IRCTC.NS", "LICHSGFIN.NS",
        "SRF.NS", "ADANIGREEN.NS", "ADANITRANS.NS", "ADANIGAS.NS"]

    MID_CAP_TICKERS = [
        "MANKIND.NS", "AUROPHARMA.NS", "BANKBARODA.NS", "CANBK.NS", "FEDERALBNK.NS", "LTFH.NS",
        "GLAND.NS", "GMRINFRA.NS", "GUJGAZ.NS", "INDIGO.NS", "PAGEIND.NS", "MPHASIS.NS",
        "DIXON.NS", "POLYCAB.NS", "VOLTAS.NS", "TVSMOTOR.NS", "BALKRISIND.NS", "CROMPTON.NS",
        "BIOCON.NS", "MAXFINANCIAL.NS", "CHOLAFIN.NS", "ICICISEC.NS", "PERSISTENT.NS",
        "MUTHOOTFIN.NS", "ASTRAL.NS", "JUBILANT.NS", "ABB.NS", "ALKEM.NS",
        "PIIND.NS", "TATAELXSI.NS", "COROMANDEL.NS", "MANAPPURAM.NS", "ADITYABIRLA.NS",
        "ALEMBIC.NS", "BATAINDIA.NS", "LTIM.NS", "GODREJPROP.NS", "DEEPAKNTR.NS",
        "TORNTPHARM.NS", "SUPREMEIND.NS", "THERMAX.NS", "RBLBANK.NS", "SUNDARMFIN.NS",
        "SHRIRAMFIN.NS", "LAURUSLABS.NS", "UBL.NS", "JSWENERGY.NS", "IDFCFIRSTB.NS",
        "IEX.NS", "ZYDUSLIFE.NS", "KEI.NS", "BLUESTARCO.NS", "TRENT.NS", "ADANIWIL.NS",
        "ADANIPOWER.NS", "GUJFLUORO.NS", "CASTROLIND.NS", "INDHOTEL.NS", "RADICO.NS",
        "APLAPOLLO.NS", "RELAXO.NS", "FINEORG.NS", "HATSUN.NS", "PRESTIGE.NS", "OBEROIRLTY.NS",
        "VARUNBEV.NS", "EMAMILTD.NS", "ZEEL.NS", "BHEL.NS", "IRCTC.NS", "CONCOR.NS",
        "IGL.NS", "JSWSTEEL.NS", "CUMMINSIND.NS", "KAJARIACER.NS", "RITES.NS",
        "SKFINDIA.NS", "HAL.NS", "CESC.NS", "ENDURANCE.NS", "GRINDWELL.NS", "GNFC.NS",
        "NAVINFLUOR.NS", "LINDEINDIA.NS", "IEX.NS", "AIAENG.NS", "KANSAINER.NS", "BIRLASOFT.NS",
        "COFORGE.NS", "SYNGENE.NS", "NARAYANA.NS", "FORTIS.NS", "INDRAPRA MEDICAL.NS",
        "PVR.NS", "INOXLEISUR.NS", "SPANDANA.NS", "CUB.NS", "KVB.NS", "SOUTHBANK.NS"]
    
    SMALL_CAP_TICKERS = [
    "LAURUSLABS.NS", "GOFORTHPHIL.NS", "DELHIVERY.NS", "ASTERDM.NS", "PIRAMALPHAR.NS",
    "APTUS.NS", "CSCCL.NS", "CITYUNION.NS", "PNCINFRA.NS", "TATACHEM.NS", "HINDCOPPER.NS",
    "NAVINFLUOR.NS", "BEML.NS", "GRSE.NS", "SCHNEIDER.NS", "GMDC.NS", "CRAG.NS",
    "CRAFTSMAN.NS", "SCHAFFLER.NS", "MAZDOCK.NS", "COCHIN.NS", "GARDENREACH.NS",
    "RVNL.NS", "IRCON.NS", "IRFC.NS", "HUDCO.NS", "ENGINERSIND.NS", "NBCC.NS", "RITES.NS",
    "TANLA.NS", "ROUTE.NS", "SUBEX.NS", "INTELLECT.NS", "HFCL.NS", "TEJASNET.NS",
    "STERLITE.NS", "DEEPAKFERT.NS", "BALRAMCHIN.NS", "DHAMPURSUG.NS", "EIDPARRY.NS",
    "AVANTIFEED.NS", "VENKY.NS", "KRBL.NS", "SOMANYCERA.NS", "ORIENTCEM.NS",
    "INDIACEM.NS", "HEIDELBERG.NS", "JINDALSAW.NS", "WELSPUNCORP.NS", "RATNAMANI.NS",
    "FINOLEXCAB.NS", "POLYCAB.NS", "BOROSIL.NS", "SWSOLAR.NS", "INOXWIND.NS",
    "INDIAGLYCO.NS", "GUJALKALI.NS", "DCW.NS", "THIRUMALAI.NS", "RUCHIRA.NS", "JKPAPER.NS",
    "WESTCOAST.NS", "DISHMAN.NS", "SEQUENT.NS", "CAPLIPOINT.NS", "GRANULES.NS",
    "NATCO.NS", "NEULAND.NS", "WELSPUNIND.NS", "RAYMOND.NS", "ARVIND.NS", "FLFL.NS",
    "HIMADRI.NS", "ORIENTCARB.NS", "PNBGILTS.NS", "UJJIVANSFB.NS", "EQUITAS.NS",
    "CSBBANK.NS", "DCBBANK.NS", "REPCOHOME.NS"]

    tickers = LARGE_CAP_TICKERS + MID_CAP_TICKERS + SMALL_CAP_TICKERS
    # fib_level_filter = [40, 60]
    # pipeline.strategy_1(tickers, fib_level_filter=fib_level_filter,start="2024-10-01",end="2025-05-01")
    # fib_level_filter = [50, 61]
    # fib_pipeline.strategy_2(tickers,fib_level_filter=fib_level_filter,  start="2023-11-01",end="2024-05-02") #period='1y'
    fib_pipeline.strategy_no_filter(tickers,  start="2023-11-01",end="2024-05-02") #period='1y'
    # gen_pipeline.multi_confluence_strategy(tickers,start="2023-11-01",end="2024-05-02")
    # gen_pipeline.strategy_bullish_trend(tickers, start="2024-01-01", end="2024-10-01")
    # gen_pipeline.strategy_bearish_trend(tickers, start="2024-01-01", end="2024-10-01")
    # gen_pipeline.strategy_neutral_breakout(tickers, start="2024-01-01", end="2024-10-01")
    # gen_pipeline.strategy_high_volatility(tickers, start="2024-01-01", end="2024-10-01")
    # gen_pipeline.strategy_volume_burst(tickers,start="2024-06-01",end="2024-09-02")



    