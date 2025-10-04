# backtest/backtester.py
import pandas as pd
from strategies.fibonacci_strategy import FibonacciStrategy

class StepBacktester:
    def __init__(self, df, strategy):
        self.df = df.copy()
        self.strategy = strategy
        self.trades = []

    def run_step_by_step(self):
        for i in range(self.strategy.fibo.lookback, len(self.df)):
            df_slice = self.df.iloc[:i+1]  # all data up to current day
            signal, last_row = self.strategy.check_trade_signal(df_slice)
            if signal:
                self.trades.append({
                    "date": last_row.name,
                    "signal": signal,
                    "price": last_row["Close"],
                    "fib_level": last_row["Fibo_Nearest_Level"]
                })
                print(f"{last_row.name.date()}: {signal} at {last_row['Close']} (Fib {last_row['Fibo_Nearest_Level']})")
        trades_df = pd.DataFrame(self.trades)
        return trades_df
