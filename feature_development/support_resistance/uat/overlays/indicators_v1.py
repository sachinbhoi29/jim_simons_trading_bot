import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class BaseOverlay:
    def plot(self, ax, df):
        raise NotImplementedError("Overlay must implement a plot method.")


class MovingAverageOverlay(BaseOverlay):
    def __init__(self, window=20, color="blue",show=True):
        self.window = window
        self.color = color
        self.show = show

    def compute(self, df):
        df[f"MA_{self.window}"] = df["Close"].rolling(self.window).mean()
        return df
    def plot(self, ax, df):
        df = self.compute(df)  # ensure values exist        
        if self.show:
            ax.plot(df.index, df[f"MA_{self.window}"], label=f"MA{self.window}", color=self.color)


class EMAOverlay(BaseOverlay):
    def __init__(self, window=20, color="orange", show=True):
        self.window = window
        self.color = color
        self.show = show

    def compute(self, df):
        df[f"EMA_{self.window}"] = df["Close"].ewm(span=self.window, adjust=False).mean()
        return df

    def plot(self, ax, df):
        df = self.compute(df)  # ensure values exist
        if self.show:
            ax.plot(df.index, df[f"EMA_{self.window}"], label=f"EMA{self.window}", color=self.color)


class RSIOverlay(BaseOverlay):
    def __init__(self, period=14, color="purple", upper=70, lower=30, subplot=False, show=True):
        self.period = period
        self.color = color
        self.upper = upper
        self.lower = lower
        self.subplot = subplot  # Optionally allow RSI to be plotted on a separate axis
        self.show = show

    def calculate_rsi(self, series):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(self.period).mean()
        avg_loss = loss.rolling(self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute(self, df):
        """Add RSI column to df and return it"""
        df[f"RSI_{self.period}"] = self.calculate_rsi(df["Close"])
        return df

    def plot(self, ax, df):
        df = self.compute(df)  # ensure RSI exists
        if self.show:
            ax.plot(df.index, df[f"RSI_{self.period}"], label=f"RSI({self.period})", color=self.color, linestyle="--")
            ax.axhline(self.upper, color="red", linestyle="dotted", linewidth=1)
            ax.axhline(self.lower, color="green", linestyle="dotted", linewidth=1)


class VolumeOverlay(BaseOverlay):
    def __init__(self, color="gray", alpha=0.4):
        self.color = color
        self.alpha = alpha

    def compute(self, df):
        """No new column needed; just return df"""
        return df

    def plot(self, ax, df):
        df = self.compute(df)  # keep consistent API
        if "Volume" not in df.columns:
            ax.text(0.5, 0.5, "No Volume Data", ha="center", va="center", transform=ax.transAxes, color="gray")
            return
        ax.bar(df.index, df["Volume"], color=self.color, alpha=self.alpha, label="Volume")
        ax.set_ylabel("Volume")



class MACDOverlay(BaseOverlay):
    def __init__(self, fast=12, slow=26, signal=9, color_macd="blue", color_signal="red", show=True):
        self.fast = fast; self.slow = slow; self.signal = signal
        self.color_macd = color_macd; self.color_signal = color_signal; self.show = show

    def compute(self, df):
        df["EMA_fast"] = df["Close"].ewm(span=self.fast, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.slow, adjust=False).mean()
        df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
        df["MACD_signal"] = df["MACD"].ewm(span=self.signal, adjust=False).mean()
        return df

    def plot(self, ax, df):
        df = self.compute(df)
        if self.show:
            ax.plot(df.index, df["MACD"], label="MACD", color=self.color_macd)
            ax.plot(df.index, df["MACD_signal"], label="Signal", color=self.color_signal)


class BollingerBandsOverlay(BaseOverlay):
    def __init__(self, window=20, num_std=2, color_mid="blue", color_band="gray", alpha=0.2, show=True):
        self.window = window; self.num_std = num_std
        self.color_mid = color_mid; self.color_band = color_band; self.alpha = alpha; self.show = show

    def compute(self, df):
        rolling_mean = df["Close"].rolling(self.window).mean()
        rolling_std = df["Close"].rolling(self.window).std()
        df[f"BB_mid_{self.window}"] = rolling_mean
        df[f"BB_upper_{self.window}"] = rolling_mean + self.num_std * rolling_std
        df[f"BB_lower_{self.window}"] = rolling_mean - self.num_std * rolling_std
        return df

    def plot(self, ax, df):
        df = self.compute(df)
        if self.show:
            ax.plot(df.index, df[f"BB_mid_{self.window}"], label=f"BB Mid {self.window}", color=self.color_mid)
            ax.fill_between(df.index, df[f"BB_lower_{self.window}"], df[f"BB_upper_{self.window}"],
                            color=self.color_band, alpha=self.alpha, label="Bollinger Bands")


class StochasticOscillatorOverlay(BaseOverlay):
    def __init__(self, k_window=14, d_window=3, color_k="orange", color_d="purple", upper=80, lower=20, show=True):
        self.k_window = k_window; self.d_window = d_window
        self.color_k = color_k; self.color_d = color_d; self.upper = upper; self.lower = lower; self.show = show

    def compute(self, df):
        low_min = df["Low"].rolling(window=self.k_window).min()
        high_max = df["High"].rolling(window=self.k_window).max()
        df["%K"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
        df["%D"] = df["%K"].rolling(window=self.d_window).mean()
        return df

    def plot(self, ax, df):
        df = self.compute(df)
        if self.show:
            ax.plot(df.index, df["%K"], label="%K", color=self.color_k)
            ax.plot(df.index, df["%D"], label="%D", color=self.color_d)
            ax.axhline(self.upper, color="red", linestyle="dotted", linewidth=1)
            ax.axhline(self.lower, color="green", linestyle="dotted", linewidth=1)


class ATROverlay(BaseOverlay):
    def __init__(self, window=14, color="brown", show=True):
        self.window = window; self.color = color; self.show = show

    def compute(self, df):
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = high_low.to_frame("hl").join(high_close.to_frame("hc")).join(low_close.to_frame("lc")).max(axis=1)
        df[f"ATR_{self.window}"] = tr.rolling(self.window).mean()
        return df

    def plot(self, ax, df):
        df = self.compute(df)
        if self.show:
            ax.plot(df.index, df[f"ATR_{self.window}"], label=f"ATR({self.window})", color=self.color)

class FibonacciOverlay(BaseOverlay):
    def __init__(self, lookback=100, levels=None, colors=None, linestyles=None, show=True):
        self.lookback = lookback
        self.levels = levels if levels is not None else [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.show = show

        self.colors = colors if colors is not None else plt.cm.viridis(np.linspace(0, 1, len(self.levels)))
        self.linestyles = linestyles if linestyles is not None else ["--"] * len(self.levels)

        # These will be cached after compute
        self.levels_dict = {}
        self.x1 = None
        self.x2 = None

    def compute(self, df):
        if df.empty or len(df) < self.lookback:
            return df

        df = df.copy()
        recent_df = df.iloc[-self.lookback:]

        swing_high_idx = recent_df["High"].idxmax()
        swing_low_idx = recent_df["Low"].idxmin()
        swing_high = df.loc[swing_high_idx, "High"]
        swing_low = df.loc[swing_low_idx, "Low"]

        # if swing_low_idx > swing_high_idx:
        #     swing_low_idx, swing_high_idx = swing_high_idx, swing_low_idx
        #     swing_low, swing_high = swing_high, swing_low

        # Always make 0% = low and 100% = high (top always 100%)
        self.x1 = swing_high_idx
        self.x2 = swing_low_idx

        # Flip: 0% = high, 100% = low
        self.levels_dict = {}
        for level in self.levels:
            price = swing_high - (swing_high - swing_low) * level
            self.levels_dict[f"{int(level * 100)}%"] = price

        # Last close analysis
        last_close = df["Close"].iloc[-1]
        distances = {k: abs(last_close - v) for k, v in self.levels_dict.items()}
        closest_level = min(distances, key=distances.get)
        closest_price = self.levels_dict[closest_level]
        percent_diff = 100 * (last_close - closest_price) / closest_price

        # Determine status
        if last_close < closest_price:
            self.fibo_status = "approaching"
        elif last_close > closest_price:
            # Check if it has crossed previously
            crossed_before = any(df["Close"].iloc[:-1] > closest_price)
            self.fibo_status = "crossed then approaching" if crossed_before else "crossed"
        else:
            self.fibo_status = "at level"

        # Store results in overlay object
        self.fibo_nearest_level = closest_level
        self.fibo_close_percent = round(percent_diff, 2)
        self.fibo_actual_level = closest_price

        # Store in DataFrame
        df.loc[df.index[-1], "Fibo_Nearest_Level"] = closest_level
        df.loc[df.index[-1], "Fibo_Percent_Offset"] = self.fibo_close_percent
        df.loc[df.index[-1], "Fibo_Status_Last_Close"] = self.fibo_status

        # Store all levels for reference
        for label, price in self.levels_dict.items():
            df.loc[df.index[-1], f"Fib_{label}"] = price

        return df

    def plot(self, ax, df):
        if not self.show or not self.levels_dict or self.x1 is None or self.x2 is None:
            return

        # Plot Fibonacci levels
        for i, (label, price) in enumerate(self.levels_dict.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            ax.plot([self.x1, self.x2], [price, price], color=color, linestyle=linestyle, label=f"{label} - {price:.2f}")
            ax.text(self.x2, price, f"{label}: {price:.2f}", color=color, fontsize=8, ha="left", va="center")

        # Last close as a star
        last_close = df["Close"].iloc[-1]
        distances = {k: abs(last_close - v) for k, v in self.levels_dict.items()}
        closest_level = min(distances, key=distances.get)

        # Place star right above the last candle
        x_pos = df.index[-1]  # last candle
        ax.scatter(x_pos, last_close, marker='*', color='black', s=100, zorder=5)

        # Label slightly above/right of the star for space
        y_offset = (df["High"].max() - df["Low"].min()) * 0.01  # 1% of price range
        ax.text(x_pos, last_close + y_offset, f" {closest_level}", color='black', fontsize=9, va="bottom", ha="center")


class FibonacciOverlayImproved(BaseOverlay):
    def __init__(self, depth=5, levels=None, colors=None, linestyles=None, show=True):
        self.depth = depth  # depth for swing detection
        self.levels = levels if levels is not None else [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.show = show

        self.colors = colors if colors is not None else plt.cm.viridis(np.linspace(0, 1, len(self.levels)))
        self.linestyles = linestyles if linestyles is not None else ["--"] * len(self.levels)

        self.levels_dict = {}
        self.x1 = None
        self.x2 = None

    def detect_swings(self, df):
        highs = df['High'].rolling(window=self.depth * 2 + 1, center=True).apply(lambda x: x[self.depth] == max(x), raw=True)
        lows = df['Low'].rolling(window=self.depth * 2 + 1, center=True).apply(lambda x: x[self.depth] == min(x), raw=True)

        swing_highs = df[highs == 1.0]
        swing_lows = df[lows == 1.0]

        return swing_highs, swing_lows

    def compute(self, df):
        if df.empty or len(df) < self.depth * 2 + 1:
            return df

        df = df.copy()
        swing_highs, swing_lows = self.detect_swings(df)

        if swing_highs.empty or swing_lows.empty:
            return df

        # Use the two most recent alternating swings (low then high or high then low)
        last_swing_high = swing_highs.iloc[-1]
        last_swing_low = swing_lows[swing_lows.index < last_swing_high.name].iloc[-1] \
            if last_swing_high.name > swing_lows.index[0] else None

        if last_swing_low is None:
            last_swing_low = swing_lows.iloc[-1]
            last_swing_high = swing_highs[swing_highs.index < last_swing_low.name].iloc[-1]

        if last_swing_high.name < last_swing_low.name:
            swing_high, swing_low = last_swing_low['High'], last_swing_high['Low']
            idx_high, idx_low = last_swing_low.name, last_swing_high.name
        else:
            swing_high, swing_low = last_swing_high['High'], last_swing_low['Low']
            idx_high, idx_low = last_swing_high.name, last_swing_low.name

        self.x1, self.x2 = idx_high, idx_low

        # Create levels
        self.levels_dict = {}
        for level in self.levels:
            price = swing_high - (swing_high - swing_low) * level
            self.levels_dict[f"{int(level * 100)}%"] = price

        # Distance from last close
        last_close = df["Close"].iloc[-1]
        distances = {k: abs(last_close - v) for k, v in self.levels_dict.items()}
        closest_level = min(distances, key=distances.get)
        closest_price = self.levels_dict[closest_level]
        percent_diff = 100 * (last_close - closest_price) / closest_price

        if last_close < closest_price:
            self.fibo_status = "approaching"
        elif last_close > closest_price:
            crossed_before = any(df["Close"].iloc[:-1] > closest_price)
            self.fibo_status = "crossed then approaching" if crossed_before else "crossed"
        else:
            self.fibo_status = "at level"

        # Store in object
        self.fibo_nearest_level = closest_level
        self.fibo_close_percent = round(percent_diff, 2)
        self.fibo_actual_level = closest_price

        # Store in DataFrame
        df.loc[df.index[-1], "Fibo_Nearest_Level"] = closest_level
        df.loc[df.index[-1], "Fibo_Percent_Offset"] = self.fibo_close_percent
        df.loc[df.index[-1], "Fibo_Status_Last_Close"] = self.fibo_status

        for label, price in self.levels_dict.items():
            df.loc[df.index[-1], f"Fib_{label}"] = price

        return df

    def plot(self, ax, df):
        if not self.show or not self.levels_dict or self.x1 is None or self.x2 is None:
            return

        for i, (label, price) in enumerate(self.levels_dict.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            ax.plot([self.x1, self.x2], [price, price], color=color, linestyle=linestyle, label=f"{label} - {price:.2f}")
            ax.text(self.x2, price, f"{label}: {price:.2f}", color=color, fontsize=8, ha="left", va="center")

        last_close = df["Close"].iloc[-1]
        distances = {k: abs(last_close - v) for k, v in self.levels_dict.items()}
        closest_level = min(distances, key=distances.get)

        x_pos = df.index[-1]
        y_offset = (df["High"].max() - df["Low"].min()) * 0.01
        ax.scatter(x_pos, last_close, marker='*', color='black', s=100, zorder=5)
        ax.text(x_pos, last_close + y_offset, f" {closest_level}", color='black', fontsize=9, va="bottom", ha="center")


class VWAPOverlay(BaseOverlay):
    def __init__(self, color="blue", show=True):
        self.color = color
        self.show = show

    def compute(self, df):
        """Add VWAP column to df"""
        if "Volume" not in df.columns:
            raise ValueError("VWAP requires 'Volume' column in DataFrame")

        # Typical price
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3

        # Cumulative calculations
        df["Cum_TPV"] = (typical_price * df["Volume"]).cumsum()
        df["Cum_Vol"] = df["Volume"].cumsum()

        # VWAP
        df["VWAP"] = df["Cum_TPV"] / df["Cum_Vol"]
        return df

    def plot(self, ax, df):
        df = self.compute(df)
        if self.show:
            ax.plot(df.index, df["VWAP"], label="VWAP", color=self.color, linewidth=1.2)

