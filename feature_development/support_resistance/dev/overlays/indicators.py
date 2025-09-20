

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
