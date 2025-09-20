# overlays/support_resistance.py

from scipy.signal import find_peaks
import numpy as np

class SupportResistanceZones:
    def __init__(self, tolerance=0.005, min_points=3, top_n=6):
        self.tolerance = tolerance
        self.min_points = min_points
        self.top_n = top_n
        self.zones = []

    def cluster_levels(self, levels):
        levels = sorted(levels)
        clusters = []
        cluster = [levels[0]]
        for level in levels[1:]:
            if abs(level - np.mean(cluster)) <= self.tolerance * np.mean(cluster):
                cluster.append(level)
            else:
                clusters.append(cluster)
                cluster = [level]
        clusters.append(cluster)
        return [(min(c), max(c), len(c)) for c in clusters if len(c) >= self.min_points]

    def find_zones(self, df):
        peaks, _ = find_peaks(df["Close"])
        troughs, _ = find_peaks(-df["Close"])
        levels = np.concatenate([df["Close"].iloc[peaks], df["Close"].iloc[troughs]])
        zones = self.cluster_levels(levels)
        self.zones = sorted(zones, key=lambda x: x[2], reverse=True)[:self.top_n]

    def plot(self, ax, df):
        self.find_zones(df)
        for i, (low, high, strength) in enumerate(self.zones, 1):
            ax.axhspan(low, high, color="orange", alpha=0.2)
            ax.text(df.index[0], (low + high) / 2, f"Z{i}", color="brown", va="center")
