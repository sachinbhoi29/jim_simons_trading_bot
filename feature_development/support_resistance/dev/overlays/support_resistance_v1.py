# overlays/support_resistance.py
import pandas as pd
from scipy.signal import find_peaks
import numpy as np


class SupportResistanceZones:
    def __init__(self, atr_window=14, min_touches=3, top_n=6, recent_weight=0.7):
        """
        :param atr_window: ATR window for adaptive tolerance
        :param min_touches: minimum touches required to validate a zone
        :param top_n: number of strongest zones to keep
        :param recent_weight: weight factor to emphasize recent touches
        """
        self.atr_window = atr_window
        self.min_touches = min_touches
        self.top_n = top_n
        self.recent_weight = recent_weight
        self.zones = []

    def compute_atr(self, df):
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        tr = np.max([high_low, high_close, low_close], axis=0)
        atr = pd.Series(tr).rolling(self.atr_window).mean()
        return atr

    def cluster_levels(self, levels, atr):
        """Cluster levels with ATR-based tolerance"""
        levels = sorted(levels)
        clusters = []
        cluster = [levels[0]]
        tol = atr.iloc[-1] * 0.5  # tighter: 0.5 * ATR
        for level in levels[1:]:
            if abs(level - np.mean(cluster)) <= tol:
                cluster.append(level)
            else:
                clusters.append(cluster)
                cluster = [level]
        clusters.append(cluster)

        return [(min(c), max(c), len(c)) for c in clusters if len(c) >= self.min_touches]

    def find_zones(self, df):
        atr = self.compute_atr(df)

        # Find swing highs and lows
        peaks, _ = find_peaks(df["High"])
        troughs, _ = find_peaks(-df["Low"])

        # Candidate levels: highs and lows
        levels = np.concatenate([df["High"].iloc[peaks], df["Low"].iloc[troughs]])

        clusters = self.cluster_levels(levels, atr)

        # Score zones by touches, recency weight
        scores = []
        for low, high, touches in clusters:
            midpoint = (low + high) / 2
            # Count how many closes are within the zone
            hits = ((df["Close"] >= low) & (df["Close"] <= high)).astype(int)

            # Apply recency weight
            weights = np.linspace(self.recent_weight, 1.0, len(df))
            strength = np.sum(hits * weights)

            scores.append((low, high, strength))

        # Keep strongest zones
        self.zones = sorted(scores, key=lambda x: x[2], reverse=True)[: self.top_n]

    def plot(self, ax, df):
        self.find_zones(df)
        for i, (low, high, strength) in enumerate(self.zones, 1):
            ax.axhspan(low, high, color="orange", alpha=0.2)
            ax.text(
                df.index[0],
                (low + high) / 2,
                f"Z{i}",
                color="brown",
                va="center",
                fontsize=9,
            )
