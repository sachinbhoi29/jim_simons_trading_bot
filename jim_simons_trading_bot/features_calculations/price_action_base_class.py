import pandas as pd

class PriceActionBase:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def compute_candlestick_patterns(self):
        df = self.df

        # Bullish Engulfing
        df['bullish_engulfing'] = ((df['Close'].shift(1) < df['Open'].shift(1)) &
                                   (df['Close'] > df['Open']) &
                                   (df['Close'] > df['Open'].shift(1)) &
                                   (df['Open'] < df['Close'].shift(1)))

        # Bearish Engulfing
        df['bearish_engulfing'] = ((df['Close'].shift(1) > df['Open'].shift(1)) &
                                   (df['Close'] < df['Open']) &
                                   (df['Close'] < df['Open'].shift(1)) &
                                   (df['Open'] > df['Close'].shift(1)))

        # Hammer
        df['hammer'] = ((df['High'] - df['Low']) > 3 * abs(df['Open'] - df['Close'])) & \
                       ((df['Close'] - df['Low']) / (1e-6 + df['High'] - df['Low']) > 0.6) & \
                       ((df['Open'] - df['Low']) / (1e-6 + df['High'] - df['Low']) > 0.6)

        # Doji
        df['doji'] = (abs(df['Close'] - df['Open']) <= (0.1 * (df['High'] - df['Low'])))

        # Inside Bar
        df['inside_bar'] = ((df['High'] < df['High'].shift(1)) &
                            (df['Low'] > df['Low'].shift(1)))

        # Outside Bar (range expansion)
        df['outside_bar'] = ((df['High'] > df['High'].shift(1)) &
                             (df['Low'] < df['Low'].shift(1)))

        # Morning Star (3-bar bullish reversal)
        df['morning_star'] = (
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            (abs(df['Close'].shift(1) - df['Open'].shift(1)) <= 0.3 * (df['High'].shift(1) - df['Low'].shift(1))) &
            (df['Close'] > ((df['Open'].shift(2) + df['Close'].shift(2)) / 2))
        )

        # Evening Star (3-bar bearish reversal)
        df['evening_star'] = (
            (df['Close'].shift(2) > df['Open'].shift(2)) &
            (abs(df['Close'].shift(1) - df['Open'].shift(1)) <= 0.3 * (df['High'].shift(1) - df['Low'].shift(1))) &
            (df['Close'] < ((df['Open'].shift(2) + df['Close'].shift(2)) / 2))
        )

        # Pin Bar (long wick rejection candle)
        df['pin_bar'] = (((df['High'] - df[["Open", "Close"]].max(axis=1)) > 
                          2 * abs(df['Open'] - df['Close'])) |
                         ((df[["Open", "Close"]].min(axis=1) - df['Low']) > 
                          2 * abs(df['Open'] - df['Close'])))

        self.df = df
        return self.df

    def compute_support_resistance(self, window=20):
        df = self.df
        df['local_support'] = df['Low'].rolling(window, center=True).min()
        df['local_resistance'] = df['High'].rolling(window, center=True).max()
        self.df = df
        return self.df

    def compute_context_flags(self, tolerance=0.01):
        """
        Adds context flags such as near support/resistance and big candle.
        """
        df = self.df
        # Price near support/resistance
        df['near_support'] = abs(df['Close'] - df['local_support']) / df['Close'] < tolerance
        df['near_resistance'] = abs(df['Close'] - df['local_resistance']) / df['Close'] < tolerance

        # Big candle: body size compared to rolling average range
        body = abs(df['Close'] - df['Open'])
        range_avg = (df['High'] - df['Low']).rolling(20).mean()
        df['big_candle'] = body > 1.5 * range_avg

        self.df = df
        return self.df