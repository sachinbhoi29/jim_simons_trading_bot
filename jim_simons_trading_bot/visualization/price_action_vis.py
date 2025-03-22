import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

class PriceActionVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def plot(self, window=100, save_path=None):
        df = self.df.tail(window).copy()
        df.index = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq='B')
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        df = df.replace([np.inf, -np.inf], np.nan)

        # Convert known pattern flags to boolean
        pattern_cols = ['bullish_engulfing', 'bearish_engulfing', 'hammer', 'doji', 'inside_bar']
        for col in pattern_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Top: Line chart with markers
        ax1.plot(df.index, df["Close"], label="Close Price", color='black')
        marker_kwargs = dict(markersize=10, linestyle='None')

        if df['bullish_engulfing'].any():
            ax1.plot(df.index[df['bullish_engulfing']], df['Close'][df['bullish_engulfing']],
                     marker='^', color='green', label='Bullish Engulfing', **marker_kwargs)
        if df['bearish_engulfing'].any():
            ax1.plot(df.index[df['bearish_engulfing']], df['Close'][df['bearish_engulfing']],
                     marker='v', color='red', label='Bearish Engulfing', **marker_kwargs)
        if df['hammer'].any():
            ax1.plot(df.index[df['hammer']], df['Close'][df['hammer']],
                     marker='H', color='orange', label='Hammer', **marker_kwargs)
        if df['doji'].any():
            ax1.plot(df.index[df['doji']], df['Close'][df['doji']],
                     marker='.', color='gray', label='Doji', **marker_kwargs)
        if df['inside_bar'].any():
            ax1.plot(df.index[df['inside_bar']], df['Close'][df['inside_bar']],
                     marker='|', color='blue', label='Inside Bar', **marker_kwargs)

        if 'local_support' in df.columns:
            ax1.plot(df.index, df['local_support'], '--', color='green', alpha=0.4, label='Support')
        if 'local_resistance' in df.columns:
            ax1.plot(df.index, df['local_resistance'], '--', color='red', alpha=0.4, label='Resistance')

        ax1.set_title("Close Price with Price Action Patterns")
        ax1.legend()
        ax1.grid(True)

        # Bottom: Candlestick chart
        candle_width = 0.6
        wick_width = 1.5

        for idx, row in df.iterrows():
            color = 'green' if row['Close'] >= row['Open'] else 'red'
            # Wick
            ax2.plot([idx, idx], [row['Low'], row['High']], color='black', linewidth=wick_width)
            # Body
            ax2.add_patch(Rectangle(
                (mdates.date2num(idx) - candle_width / 2, min(row['Open'], row['Close'])),
                candle_width,
                abs(row['Close'] - row['Open']),
                color=color,
                linewidth=0
            ))

        ax2.set_title("Candlestick Chart")
        ax2.grid(True)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Chart saved to {save_path}")
        # else:
        #     plt.show()