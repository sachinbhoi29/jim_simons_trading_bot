# candlestick_chart.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from screeninfo import get_monitors


class CandlestickChart:
    def __init__(self, df, ticker="Ticker", dpi=100, show_candles=True, show=True):
        self.df = df
        self.ticker = ticker
        self.dpi = dpi
        self.show_candles = show_candles
        self.overlays = []
        self.subplots = []  # Stores (overlay, height_ratio)
        self.show = show

    def add_overlay(self, overlay):
        self.overlays.append(overlay)

    def only_df(self):
        """
        Compute all overlays/subplots and return the enriched DataFrame,
        without plotting anything.
        """
        df = self.df.copy()

        # overlays
        for overlay in self.overlays:
            if hasattr(overlay, "compute"):  # overlays should implement compute(df)
                df = overlay.compute(df)

        # subplots
        for overlay, _ in self.subplots:
            if hasattr(overlay, "compute"):
                df = overlay.compute(df)

        return df

    def add_subplot(self, overlay, height_ratio=1):
        self.subplots.append((overlay, height_ratio))

    def plot(self):
        df = self.df.reset_index()
        df['Date'] = df['Date'].map(mdates.date2num)
        ohlc = df[['Date', 'Open', 'High', 'Low', 'Close']].values

        if not self.show:
            return df  # return only data

        # Plot config
        height_ratios = [4] + [r for _, r in self.subplots]
        total_ratio = sum(height_ratios)
        base_height = 6  # main chart height
        height_in = base_height * total_ratio / 4
        width_in = 16  # good default width

        fig, axs = plt.subplots(
            len(height_ratios), 1, sharex=True,
            figsize=(width_in, height_in), dpi=self.dpi,
            gridspec_kw={"height_ratios": height_ratios}
        )

        if len(height_ratios) == 1:
            axs = [axs]

        ax_main = axs[0]
        if self.show_candles:
            candlestick_ohlc(ax_main, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.8)

        for overlay in self.overlays:
            overlay.plot(ax_main, self.df)

        ax_main.xaxis_date()
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_main.set_title(f"{self.ticker} - Candlestick Chart", fontsize=20)
        ax_main.set_ylabel("Price")
        ax_main.grid(True)
        ax_main.legend()
        plt.setp(ax_main.get_xticklabels(), rotation=45, ha="right")
        plt.subplots_adjust(bottom=0.15)

        # Subplots
        for i, (overlay, _) in enumerate(self.subplots):
            ax = axs[i + 1]
            overlay.plot(ax, self.df)
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()


