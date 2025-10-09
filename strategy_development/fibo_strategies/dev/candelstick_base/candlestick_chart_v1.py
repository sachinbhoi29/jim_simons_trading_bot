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

    def add_text(self, text, x=0.01, y=0.99, fontsize=10, color='black', bbox=True):
        """
        Store text annotation to be plotted on chart.
        x, y are in axes fraction (0-1).
        """
        self.note_text_info = {
            "text": text,
            "x": x,
            "y": y,
            "fontsize": fontsize,
            "color": color,
            "bbox": bbox
        }

    def plot(self,save_path=False):
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

        # --- Add text annotation if exists ---
        if hasattr(self, "note_text_info"):
            info = self.note_text_info
            bbox_props = dict(facecolor='yellow', alpha=0.4, boxstyle='round,pad=0.3') if info['bbox'] else None
            ax_main.text(
                info['x'], info['y'], info['text'],
                transform=ax_main.transAxes,
                fontsize=info['fontsize'],
                color=info['color'],
                verticalalignment='top',
                bbox=bbox_props
            )

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
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()



