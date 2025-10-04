# strategy/fibonacci_strategy.py
from overlays.indicators_v1 import FibonacciOverlay
from utilities.fib_utils import last_price_fib_info

class FibonacciStrategy:
    def __init__(self, lookback=50):
        self.fibo = FibonacciOverlay(lookback=lookback)

    def check_trade_signal(self, df):
        """
        Given a df with OHLC data (up to current day),
        compute indicators and return trade signal if any.
        """
        df = self.fibo.compute(df)
        last_row = df.iloc[-1]


        fib_info = last_price_fib_info(df)
        fib_percent = fib_info["fib_percent"]          # numeric, e.g., 61.8
        closest_level_price = fib_info["closest_fib_level"]  # exact price
        status = last_row.get("Fibo_Status_Last_Close")      # "approaching", "crossed", etc.

        signal = None

        # Example strategy logic
        # Buy if last price is within 50–65% retracement and approaching
        if 50 <= fib_percent <= 65 and status == "approaching":
            signal = "buy"
        # Sell if last price is within 50–65% retracement and crossed above
        elif 50 <= fib_percent <= 65 and status == "crossed":
            signal = "sell"

        return signal, last_row
