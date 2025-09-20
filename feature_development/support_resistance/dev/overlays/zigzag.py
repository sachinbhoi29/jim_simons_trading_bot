# overlays/zigzag.py

from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

class Zigzag:
    def __init__(self, distance=5, prominence=10):
        self.distance = distance
        self.prominence = prominence
        self.peaks = []
        self.troughs = []
        self.zigzag_idx = []

    def find_zigzag_points(self, close):
        self.peaks, _ = find_peaks(close, distance=self.distance, prominence=self.prominence)
        self.troughs, _ = find_peaks(-close, distance=self.distance, prominence=self.prominence)
        self.zigzag_idx = sorted(list(self.peaks) + list(self.troughs))

    def plot(self, ax, df):
        close = df["Close"].to_numpy().ravel()
        self.find_zigzag_points(close)

        ax.plot(df.index, close, color="black", alpha=0.6, label="Close")
        ax.plot(df.index[self.zigzag_idx], close[self.zigzag_idx], color="blue", linewidth=1.5, label="Zigzag")
        ax.scatter(df.index[self.peaks], close[self.peaks], color="red", marker="o", label="Peaks")
        ax.scatter(df.index[self.troughs], close[self.troughs], color="green", marker="o", label="Troughs")
