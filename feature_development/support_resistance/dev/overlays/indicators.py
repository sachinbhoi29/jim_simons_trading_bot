

class BaseOverlay:
    def plot(self, ax, df):
        raise NotImplementedError("Overlay must implement a plot method.")


class MovingAverageOverlay(BaseOverlay):
    def __init__(self, window=20, color="blue"):
        self.window = window
        self.color = color

    def plot(self, ax, df):
        df[f"MA_{self.window}"] = df["Close"].rolling(self.window).mean()
        ax.plot(df.index, df[f"MA_{self.window}"], label=f"MA{self.window}", color=self.color)        


class RSIOverlay(BaseOverlay):
    def __init__(self, period=14, color="purple", upper=70, lower=30, subplot=False):
        self.period = period
        self.color = color
        self.upper = upper
        self.lower = lower
        self.subplot = subplot  # Optionally allow RSI to be plotted on a separate axis

    def calculate_rsi(self, series):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(self.period).mean()
        avg_loss = loss.rolling(self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def plot(self, ax, df):
        rsi = self.calculate_rsi(df["Close"])
        df[f"RSI_{self.period}"] = rsi

        # Option 1: Plot RSI on the same axis
        ax.plot(df.index, rsi, label=f"RSI({self.period})", color=self.color, linestyle="--")
        ax.axhline(self.upper, color="red", linestyle="dotted", linewidth=1)
        ax.axhline(self.lower, color="green", linestyle="dotted", linewidth=1)
