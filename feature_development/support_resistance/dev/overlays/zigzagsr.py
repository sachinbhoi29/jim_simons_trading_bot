from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

class ZigzagSR:
    def __init__(self, distance=5, prominence=10, show_zigzag=True, show_peaks=True, show_troughs=True, show_zones=True,
                 tolerance=0.005, min_points=3, top_n=6, color_close="black", color_zigzag="blue", color_peaks="red",
                 color_troughs="green", color_zone="orange", alpha_zone=0.25, max_width_zone=0.01, min_width_zone=0.001, show=True):
        self.show=show; self.max_width_zone=max_width_zone; self.min_width_zone=min_width_zone
        self.distance=distance; self.prominence=prominence; self.show_zigzag=show_zigzag; self.show_peaks=show_peaks; self.show_troughs=show_troughs
        self.show_zones=show_zones; self.tolerance=tolerance; self.min_points=min_points; self.top_n=top_n
        self.color_close=color_close; self.color_zigzag=color_zigzag; self.color_peaks=color_peaks; self.color_troughs=color_troughs
        self.color_zone=color_zone; self.alpha_zone=alpha_zone
        self.peaks=[]; self.troughs=[]; self.zigzag_idx=[]; self.zones=[]

    def find_zigzag_points(self, close):
        self.peaks,_=find_peaks(close, distance=self.distance, prominence=self.prominence)
        self.troughs,_=find_peaks(-close, distance=self.distance, prominence=self.prominence)
        self.zigzag_idx=sorted(list(self.peaks)+list(self.troughs))

    @staticmethod
    def cluster_levels(levels, tolerance=0.01, min_points=2):
        if len(levels)==0: return []
        levels=sorted(levels); clusters=[]; cluster=[levels[0]]
        for level in levels[1:]:
            if abs(level-np.mean(cluster))<=tolerance*np.mean(cluster): cluster.append(level)
            else: clusters.append(cluster); cluster=[level]
        clusters.append(cluster)
        return [(min(c), max(c), len(c)) for c in clusters if len(c)>=min_points]

    def find_zones(self, close):
        all_levels=np.concatenate([close[self.peaks], close[self.troughs]])
        zones=self.cluster_levels(all_levels, tolerance=self.tolerance, min_points=self.min_points)
        filtered=[(low, high, strength) for (low, high, strength) in zones if (high-low)/low<self.max_width_zone and (high-low)/low>self.min_width_zone]
        self.zones=sorted(filtered, key=lambda x:x[2], reverse=True)[:self.top_n]

    def compute(self, df):
        if "Close" not in df.columns: raise ValueError("DataFrame must contain a 'Close' column.")
        close=df["Close"].to_numpy().ravel()
        self.find_zigzag_points(close); self.find_zones(close)

        # Zigzag column
        df["Zigzag"]=np.nan; df.loc[df.index[self.zigzag_idx], "Zigzag"]=df["Close"].iloc[self.zigzag_idx]

        # Peaks and troughs
        df["Peak"]=np.nan; df.loc[df.index[self.peaks], "Peak"]=df["Close"].iloc[self.peaks]
        df["Trough"]=np.nan; df.loc[df.index[self.troughs], "Trough"]=df["Close"].iloc[self.troughs]

        # Zones: add as low/high columns
        for i,(low,high,strength) in enumerate(self.zones,1):
            df[f"Zone{i}_low"]=low; df[f"Zone{i}_high"]=high

        return df

    def plot(self, ax, df):
        if not self.show: return df
        if "Close" not in df.columns: raise ValueError("DataFrame must contain a 'Close' column.")

        # Always compute before plotting
        df=self.compute(df); close=df["Close"].to_numpy().ravel()

        ax.plot(df.index, close, color=self.color_close, alpha=0.6, label="Close")
        if self.show_zigzag: ax.plot(df.index[self.zigzag_idx], close[self.zigzag_idx], color=self.color_zigzag, linewidth=1.5, label="Zigzag")
        if self.show_peaks: ax.scatter(df.index[self.peaks], close[self.peaks], color=self.color_peaks, marker="o", label="Peaks")
        if self.show_troughs: ax.scatter(df.index[self.troughs], close[self.troughs], color=self.color_troughs, marker="o", label="Troughs")
        if self.show_zones:
            for i,(low,high,strength) in enumerate(self.zones,1):
                ax.axhspan(low, high, color=self.color_zone, alpha=self.alpha_zone)
                ax.text(df.index[0], (low+high)/2, f"Z{i} ({strength})", color="brown", va="center", fontsize=8)
