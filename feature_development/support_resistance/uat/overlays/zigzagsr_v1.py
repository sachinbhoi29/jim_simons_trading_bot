from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

class ZigzagSR:
    '''
    | Parameter Name              | Description                                                                 |
    | -------------------------- | --------------------------------------------------------------------------- |
    | `min_peak_distance`        | Minimum number of data points between peaks/troughs                         |
    | `min_peak_prominence`      | Minimum height difference to qualify as a peak/trough                       |
    | `zone_merge_tolerance`     | Percent range within which to group levels into a zone                      |
    | `min_zone_points`          | Minimum number of touches to form a support/resistance zone                 |
    | `max_zones`                | Number of strongest zones to keep                                           |
    | `max_zone_width_ratio`     | Maximum width of a zone (as % of price level)                               |
    | `min_zone_width_ratio`     | Minimum width of a zone (as % of price level)                               |
    | `show`                     | Whether to show the plot at all (set to `True` or `False`)                  |
    | `show_zigzag_line`         | Whether to plot the zigzag lines                                            |
    | `show_peak_markers`        | Whether to plot the peaks                                                   |
    | `show_trough_markers`      | Whether to plot the troughs                                                 |
    | `show_support_resistance_zones` | Whether to show support/resistance zones                               |
    '''

    def __init__(self, min_peak_distance=5, min_peak_prominence=10, show_zigzag_line=True, show_peak_markers=True, show_trough_markers=True, show_support_resistance_zones=True,
                 zone_merge_tolerance=0.005, min_zone_points=3, max_zones=6, color_close="black", color_zigzag="blue", color_peaks="red",
                 color_troughs="green", color_zone="orange", alpha_zone=0.25, max_zone_width_ratio=0.01, min_zone_width_ratio=0.001, show=True):
        
        self.show = show
        self.max_zone_width_ratio = max_zone_width_ratio
        self.min_zone_width_ratio = min_zone_width_ratio
        self.min_peak_distance = min_peak_distance
        self.min_peak_prominence = min_peak_prominence
        self.show_zigzag_line = show_zigzag_line
        self.show_peak_markers = show_peak_markers
        self.show_trough_markers = show_trough_markers
        self.show_support_resistance_zones = show_support_resistance_zones
        self.zone_merge_tolerance = zone_merge_tolerance
        self.min_zone_points = min_zone_points
        self.max_zones = max_zones
        self.color_close = color_close
        self.color_zigzag = color_zigzag
        self.color_peaks = color_peaks
        self.color_troughs = color_troughs
        self.color_zone = color_zone
        self.alpha_zone = alpha_zone
        self.peaks = []
        self.troughs = []
        self.zigzag_idx = []
        self.zones = []

    def find_zigzag_points(self, close):
        self.peaks, _ = find_peaks(close, distance=self.min_peak_distance, prominence=self.min_peak_prominence)
        self.troughs, _ = find_peaks(-close, distance=self.min_peak_distance, prominence=self.min_peak_prominence)
        self.zigzag_idx = sorted(list(self.peaks) + list(self.troughs))

    @staticmethod
    def cluster_levels(levels, tolerance=0.01, min_points=2):
        if len(levels) == 0: return []
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
        return [(min(c), max(c), len(c)) for c in clusters if len(c) >= min_points]

    def find_zones(self, close):
        all_levels = np.concatenate([close[self.peaks], close[self.troughs]])
        zones = self.cluster_levels(all_levels, tolerance=self.zone_merge_tolerance, min_points=self.min_zone_points)
        filtered = [
            (low, high, strength)
            for (low, high, strength) in zones
            if (high - low) / low < self.max_zone_width_ratio and (high - low) / low > self.min_zone_width_ratio
        ]
        self.zones = sorted(filtered, key=lambda x: x[2], reverse=True)[:self.max_zones]

    def compute(self, df):
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain a 'Close' column.")
        close = df["Close"].to_numpy().ravel()
        self.find_zigzag_points(close)
        self.find_zones(close)

        df["Zigzag"] = np.nan
        df.loc[df.index[self.zigzag_idx], "Zigzag"] = df["Close"].iloc[self.zigzag_idx]

        df["Peak"] = np.nan
        df.loc[df.index[self.peaks], "Peak"] = df["Close"].iloc[self.peaks]
        df["Trough"] = np.nan
        df.loc[df.index[self.troughs], "Trough"] = df["Close"].iloc[self.troughs]

        for i, (low, high, strength) in enumerate(self.zones, 1):
            df[f"Zone{i}_low"] = low
            df[f"Zone{i}_high"] = high

        return df

    def plot(self, ax, df):
        if not self.show:
            return df
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain a 'Close' column.")

        df = self.compute(df)
        close = df["Close"].to_numpy().ravel()

        ax.plot(df.index, close, color=self.color_close, alpha=0.6, label="Close")
        if self.show_zigzag_line:
            ax.plot(df.index[self.zigzag_idx], close[self.zigzag_idx], color=self.color_zigzag, linewidth=1.5, label="Zigzag")
        if self.show_peak_markers:
            ax.scatter(df.index[self.peaks], close[self.peaks], color=self.color_peaks, marker="o", label="Peaks")
        if self.show_trough_markers:
            ax.scatter(df.index[self.troughs], close[self.troughs], color=self.color_troughs, marker="o", label="Troughs")
        if self.show_support_resistance_zones:
            for i, (low, high, strength) in enumerate(self.zones, 1):
                ax.axhspan(low, high, color=self.color_zone, alpha=self.alpha_zone)
                ax.text(df.index[0], (low + high) / 2, f"Z{i} ({strength})", color="brown", va="center", fontsize=8)
