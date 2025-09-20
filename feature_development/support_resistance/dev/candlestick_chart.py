# candlestick_chart.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from screeninfo import get_monitors


class CandlestickChart:
    def __init__(self, df, ticker="Ticker", dpi=100, show_candles=True):
        self.df = df
        self.ticker = ticker
        self.dpi = dpi
        self.show_candles = show_candles
        self.overlays = []

    def add_overlay(self, overlay):
        self.overlays.append(overlay)

    def plot(self):
        df = self.df.reset_index()
        df['Date'] = df['Date'].map(mdates.date2num)
        ohlc = df[['Date', 'Open', 'High', 'Low', 'Close']].values

        # Fullscreen figure
        monitor = get_monitors()[0]
        width_in = monitor.width / self.dpi
        height_in = monitor.height / self.dpi
        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=self.dpi)

        if self.show_candles:
            candlestick_ohlc(ax, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.8)

        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.subplots_adjust(bottom=0.2)

        for overlay in self.overlays:
            overlay.plot(ax, self.df)

        ax.set_title(f"{self.ticker} - Candlestick Chart", fontsize=20)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend()
        plt.show()
