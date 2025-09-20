# main.py

import yfinance as yf
from candlestick_chart import CandlestickChart
from overlays.support_resistance import SupportResistanceZones
from overlays.indicators import MovingAverageOverlay
from overlays.zigzag import Zigzag


if __name__ == "__main__":
    df = yf.download("^NSEI", period="6mo", interval="1d", auto_adjust=False, multi_level_index=False)
    df.columns.name = None

    chart = CandlestickChart(df, ticker="^NSEI", show_candles=False)
    chart.add_overlay(SupportResistanceZones())
    chart.add_overlay(MovingAverageOverlay(window=20, color="blue"))
    chart.add_overlay(MovingAverageOverlay(window=50, color="red"))
    chart.add_overlay(Zigzag())
    chart.plot()



