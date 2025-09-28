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
                 color_troughs="green", color_zone="orange", alpha_zone=0.25, max_zone_width_ratio=0.01, min_zone_width_ratio=0.001, show=True,
                 show_fibo = True,show_trendline=True,show_only_latest_fibo=True):
        self.show_only_latest_fibo = show_only_latest_fibo
        self.show_trendline=show_trendline
        self.show_fibo = show_fibo
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

    def compute_fib_levels(self, df, levels=None, min_move_pct=0.01):
        """
        Compute Fibonacci retracement levels for significant ZigZag legs only.

        Parameters:
        - df: DataFrame with 'Close'
        - levels: list of Fibonacci levels (default [0,0.236,0.382,0.5,0.618,0.786,1])
        - min_move_pct: minimum price change (%) to consider the leg significant
        """
        if levels is None:
            levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]

        fib_data = []
        close = df["Close"].to_numpy().ravel()
        for i in range(1, len(self.zigzag_idx)):
            idx1, idx2 = self.zigzag_idx[i-1], self.zigzag_idx[i]
            price1, price2 = close[idx1], close[idx2]

            # Compute absolute and relative move
            move = abs(price2 - price1)
            move_pct = move / price1

            if move_pct < min_move_pct:
                continue  # skip insignificant moves

            # Up leg
            if price2 > price1:
                fibs = {f"{int(l*100)}%": price2 - l*move for l in levels}
            else:  # Down leg
                fibs = {f"{int(l*100)}%": price2 + l*move for l in levels}

            fib_data.append({
                "start_idx": idx1, "end_idx": idx2,
                "start_price": price1, "end_price": price2,
                "fibs": fibs
            })

        self.fib_levels = fib_data
        return fib_data


    def plot_zigzag_midline(self, ax, df, zigzag_idx, close, color="magenta", linestyle="--", linewidth=2):
        """
        Plot slanted trendlines connecting the midpoints of each ZigZag leg.
        """
        if len(zigzag_idx) < 2:
            return  # Not enough points

        mid_x = []
        mid_y = []

        for i in range(1, len(zigzag_idx)):
            idx1, idx2 = zigzag_idx[i-1], zigzag_idx[i]
            price1, price2 = close[idx1], close[idx2]

            # Midpoint
            x_mid = df.index[idx1] + (df.index[idx2] - df.index[idx1]) / 2
            y_mid = (price1 + price2) / 2

            mid_x.append(x_mid)
            mid_y.append(y_mid)

        # Connect midpoints with slanted line
        ax.plot(mid_x, mid_y, color=color, linestyle=linestyle, linewidth=linewidth, label="_nolegend_")


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

        # --- Fib plotting ---
        if self.show_fibo:
            fib_data = self.compute_fib_levels(df)
            for fib_leg in fib_data:
                if self.show_only_latest_fibo:
                    fib_leg = fib_data[-1]  # pick the last leg
                idx1, idx2 = fib_leg["start_idx"], fib_leg["end_idx"]
                for label, price in fib_leg["fibs"].items():
                    ax.hlines(price, df.index[idx1], df.index[idx2],
                            colors="purple", linestyles="dotted", linewidth=1)
                    ax.text(df.index[idx2], price, f"{label}", fontsize=7,
                            color="purple", va="center")
        
        # --- Fib plotting (only latest leg) ---
        if self.show_only_latest_fibo:
            fib_data = self.compute_fib_levels(df)
            if fib_data:  # make sure there is at least one leg
                fib_leg = fib_data[-1]  # pick the last leg
                idx1, idx2 = fib_leg["start_idx"], fib_leg["end_idx"]
                for label, price in fib_leg["fibs"].items():
                    ax.hlines(price, df.index[idx1], df.index[idx2],
                            colors="purple", linestyles="dotted", linewidth=1)
                    ax.text(df.index[idx2], price, f"{label}", fontsize=7,
                            color="purple", va="center")

        # --- Trendline plotting ---
        if self.show_trendline:
            self.plot_zigzag_midline(ax, df, self.zigzag_idx, df["Close"].to_numpy(), color="magenta", linestyle="--", linewidth=2)
