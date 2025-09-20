
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

class ZigzagSR:
    def __init__(self, df,ticker):
        self.df = df
        self.ticker = ticker
        self.close = None
        self.peaks = None
        self.troughs = None
        self.zigzag_idx = None
        self.zones = None

    def plot_zigzag(self, distance=5, prominence=None, height=None, width=None, show=True):
        """Plot Zigzag (peaks & troughs) for a stock/index from Yahoo Finance.""" 
        self.df.dropna(inplace=True)

        self.close = self.df["Close"].to_numpy().ravel()
        self.peaks, _ = find_peaks(
            self.close, distance=distance, prominence=prominence,
            height=height, width=width
        )
        self.troughs, _ = find_peaks(
            -self.close, distance=distance, prominence=prominence,
            height=height, width=width
        )

        self.zigzag_idx = sorted(list(self.peaks) + list(self.troughs))

        if show:
            plt.figure(figsize=(14, 6))
            plt.plot(self.df.index, self.close, color="black", alpha=0.6, label="Close")
            plt.plot(self.df.index[self.zigzag_idx], self.close[self.zigzag_idx],
                     color="blue", linewidth=1.5, label="Zigzag")
            plt.scatter(self.df.index[self.peaks], self.close[self.peaks],
                        color="red", marker="o", label="Peaks")
            plt.scatter(self.df.index[self.troughs], self.close[self.troughs],
                        color="green", marker="o", label="Troughs")
            plt.legend()
            plt.title(f"{self.ticker} Zigzag (Crests & Troughs)")
            plt.show()

        return self.df, self.peaks, self.troughs, self.close, self.zigzag_idx

    @staticmethod
    def cluster_levels(levels, tolerance=0.01, min_points=2):
        """
        Cluster nearby price levels into support/resistance zones.
        Returns list of (low, high, strength).
        """
        if len(levels) == 0:
            return []

        levels = sorted(levels)
        clusters = []
        cluster = [levels[0]]

        for level in levels[1:]:
            if abs(level - np.mean(cluster)) <= tolerance * np.mean(cluster):
                cluster.append(level)
            else:
                clusters.append(cluster)
                cluster = [level]
        clusters.append(cluster)

        # Convert to zones with strength
        zones = [(min(c), max(c), len(c)) for c in clusters if len(c) >= min_points]
        return zones

    def find_zones(self, tolerance=0.005, min_points=3, top_n=6):
        """Find and rank merged support & resistance zones."""
        all_levels = np.concatenate([self.close[self.peaks], self.close[self.troughs]])
        zones = self.cluster_levels(all_levels, tolerance=tolerance, min_points=min_points)
        zones = sorted(zones, key=lambda x: x[2], reverse=True)[:top_n]
        self.zones = zones
        return zones

    def plot_zones(self):
        """Plot support & resistance zones along with zigzag."""
        if self.df is None or self.zones is None:
            raise ValueError("Run plot_zigzag() and find_zones() before plotting zones.")

        plt.figure(figsize=(14, 6))
        plt.plot(self.df.index, self.close, color="black", alpha=0.6, label="Close")
        plt.plot(self.df.index[self.zigzag_idx], self.close[self.zigzag_idx],
                 color="blue", linewidth=1.5, label="Zigzag")
        plt.scatter(self.df.index[self.peaks], self.close[self.peaks],
                    color="red", marker="o", label="Peaks")
        plt.scatter(self.df.index[self.troughs], self.close[self.troughs],
                    color="green", marker="o", label="Troughs")

        # for low, high, strength in self.zones:
        #     plt.axhspan(low, high, color="orange", alpha=0.25)
        #     plt.text(self.df.index[0], (low + high) / 2,
        #              f"Z({strength})", color="brown", va="center")
        for i, (low, high, strength) in enumerate(self.zones, 1):
            plt.axhspan(low, high, color="orange", alpha=0.25)
            plt.text(self.df.index[0], (low + high) / 2,
                     f"Z{i} ({strength})", color="brown", va="center")


        plt.legend()
        plt.title(f"{self.ticker} - Merged Support & Resistance Zones")
        plt.show()


# ---------------- Example Usage ----------------
df = yf.download("^NSEI", period="1y", interval="1d",
            group_by="ticker", multi_level_index=False)
zigzag = ZigzagSR(df,"^NSEI")
zigzag.plot_zigzag( prominence=10)  # adjust distance & prominence
zigzag.find_zones()  # wider tolerance, stronger zones
zigzag.plot_zones()


