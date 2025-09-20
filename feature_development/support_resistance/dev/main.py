# main.py

import yfinance as yf
from candlestick_chart import CandlestickChart
from overlays.indicators import MovingAverageOverlay, RSIOverlay,VolumeOverlay,MACDOverlay,BollingerBandsOverlay,StochasticOscillatorOverlay,ATROverlay,EMAOverlay,FibonacciOverlay
from overlays.zigzagsr import ZigzagSR


if __name__ == "__main__":
    ticker = "^NSEI"
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, multi_level_index=False)


    chart = CandlestickChart(df, ticker="NSEI", show_candles=True,show=True)
    # chart.add_overlay(MovingAverageOverlay(window=20, color="blue",show=True))
    # chart.add_overlay(MovingAverageOverlay(window=50, color="red",show=True))
    chart.add_overlay(EMAOverlay(window=50, color="red",show=True))
    chart.add_overlay(FibonacciOverlay(lookback=50, show=True))
    # chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
    # chart.add_subplot(MACDOverlay())
    # chart.add_overlay(BollingerBandsOverlay())
    # chart.add_subplot(StochasticOscillatorOverlay())
    # chart.add_subplot(ATROverlay())
    # zigzag = ZigzagSR(min_peak_distance=8, min_peak_prominence=10, zone_merge_tolerance=0.007, max_zones=8, color_zone="green", alpha_zone=0.2, show=True)
    # chart.add_overlay(zigzag)
    # chart.add_subplot(VolumeOverlay(), height_ratio=1)
    print(chart.only_df())
    chart.plot()



