

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