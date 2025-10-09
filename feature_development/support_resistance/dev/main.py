# main.py

import yfinance as yf
from candlestick_chart_v1 import CandlestickChart
from overlays.indicators_v1 import MovingAverageOverlay, RSIOverlay,VolumeOverlay,MACDOverlay,BollingerBandsOverlay,StochasticOscillatorOverlay,ATROverlay,EMAOverlay,FibonacciOverlay,VWAPOverlay
# from overlays.support_resistance import SupportResistanceZones # added in zigzagsr
from overlays.zigzagsr_v1 import ZigzagSR
from overlays.regime_detection_v1 import EnhancedRegimeOverlay


if __name__ == "__main__":
    ticker = "^NSEBANK" #NSEBANK, "^NSEI"
    df = yf.download(ticker,start="2024-01-01",end="2025-04-08",interval="1d",auto_adjust=False,multi_level_index=False)
    # df = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, multi_level_index=False)
    chart = CandlestickChart(df, ticker="NSEI", show_candles=True,show=True)
    # chart.add_overlay(MovingAverageOverlay(window=20, color="blue",show=True))
    # chart.add_overlay(EnhancedRegimeOverlay(show=True))
    # chart.add_overlay(ZigzagSR())
    # chart.add_overlay(MovingAverageOverlay(window=50, color="red",show=True))
    chart.add_overlay(EMAOverlay(window=50, color="red",show=True))
    chart.add_overlay(EMAOverlay(window=20, color="Green",show=True))
    chart.add_overlay(FibonacciOverlay(lookback=50, show=True))
    # chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
    # chart.add_subplot(MACDOverlay())
    # chart.add_overlay(BollingerBandsOverlay())
    # chart.add_subplot(StochasticOscillatorOverlay())
    # chart.add_subplot(ATROverlay())
    # chart.add_subplot(VWAPOverlay())
    zigzag = ZigzagSR(min_peak_distance=8, min_peak_prominence=10, zone_merge_tolerance=0.007, max_zones=8, color_zone="green", alpha_zone=0.2, show=True,show_fibo = False,show_trendline=False,show_only_latest_fibo=False)
    chart.add_overlay(zigzag)
    # chart.add_subplot(VolumeOverlay(), height_ratio=1)
    print(chart.only_df())
    chart.plot()





# All indicators for reference
# if __name__ == "__main__":
#     ticker = "^NSEBANK" #NSEBANK, "^NSEI"
#     df = yf.download(ticker,start="2024-01-01",end="2025-04-08",interval="1d",auto_adjust=False,multi_level_index=False)
#     # df = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, multi_level_index=False)
#     chart = CandlestickChart(df, ticker="NSEI", show_candles=True,show=True)
#     # chart.add_overlay(MovingAverageOverlay(window=20, color="blue",show=True))
#     # chart.add_overlay(EnhancedRegimeOverlay(show=True))
#     # chart.add_overlay(ZigzagSR())
#     # chart.add_overlay(MovingAverageOverlay(window=50, color="red",show=True))
#     chart.add_overlay(EMAOverlay(window=50, color="red",show=True))
#     chart.add_overlay(EMAOverlay(window=20, color="Green",show=True))
#     chart.add_overlay(FibonacciOverlay(lookback=50, show=True))
#     # chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
#     # chart.add_subplot(MACDOverlay())
#     # chart.add_overlay(BollingerBandsOverlay())
#     # chart.add_subplot(StochasticOscillatorOverlay())
#     # chart.add_subplot(ATROverlay())
#     # chart.add_subplot(VWAPOverlay())
#     zigzag = ZigzagSR(min_peak_distance=8, min_peak_prominence=10, zone_merge_tolerance=0.007, max_zones=8, color_zone="green", alpha_zone=0.2, show=True,show_fibo = False,show_trendline=False,show_only_latest_fibo=False)
#     chart.add_overlay(zigzag)
#     # chart.add_subplot(VolumeOverlay(), height_ratio=1)
#     print(chart.only_df())
#     chart.plot()
