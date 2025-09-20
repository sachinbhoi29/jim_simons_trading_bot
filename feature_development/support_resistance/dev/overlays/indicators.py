

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
