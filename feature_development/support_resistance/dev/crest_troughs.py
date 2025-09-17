import yfinance as yf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

# Download NIFTY 50 data (flat columns)
ticker = "^NSEI"
df = yf.download(ticker, period="1y", interval="1d", group_by="ticker", multi_level_index=False)

# Ensure no missing data
df.dropna(inplace=True)

# Extract 1D close price array
close = df["Close"].to_numpy().ravel()

# Detect peaks (crests) and troughs
peaks, _ = find_peaks(close, distance=5)
troughs, _ = find_peaks(-close, distance=5)

# Plot
plt.figure(figsize=(14,6))
plt.plot(df.index, close, color="black", alpha=0.6, label="Close")

zigzag_idx = sorted(list(peaks) + list(troughs))
plt.plot(df.index[zigzag_idx], close[zigzag_idx], color="blue", linewidth=1.5, label="Zigzag")

plt.scatter(df.index[peaks], close[peaks], color="red", marker="o", label="Peaks")
plt.scatter(df.index[troughs], close[troughs], color="green", marker="o", label="Troughs")

plt.legend()
plt.title("NIFTY 50 Zigzag (Crests & Troughs)")
plt.show()



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


# ---------------- Control parameters ----------------
tolerance = 0.005   # 0.5% band
min_points = 3      # at least 3 touches
top_n = 6           # limit to top 6 zones

# Combine both peaks and troughs into one set of levels
all_levels = np.concatenate([close[peaks], close[troughs]])

# Cluster both together
zones = cluster_levels(all_levels, tolerance=tolerance, min_points=min_points)

# Rank by strength & take top N
zones = sorted(zones, key=lambda x: x[2], reverse=True)[:top_n]

# ---------------- Plot ----------------
plt.figure(figsize=(14,6))
plt.plot(df.index, close, color="black", alpha=0.6, label="Close")
plt.plot(df.index[zigzag_idx], close[zigzag_idx], color="blue", linewidth=1.5, label="Zigzag")
plt.scatter(df.index[peaks], close[peaks], color="red", marker="o", label="Peaks")
plt.scatter(df.index[troughs], close[troughs], color="green", marker="o", label="Troughs")

# Draw merged zones (single band for both support & resistance)
for low, high, strength in zones:
    plt.axhspan(low, high, color="orange", alpha=0.25)
    plt.text(df.index[0], (low+high)/2, f"Z({strength})", color="brown", va="center")

plt.legend()
plt.title("NIFTY 50 - Merged Support & Resistance Zones")
plt.show()
