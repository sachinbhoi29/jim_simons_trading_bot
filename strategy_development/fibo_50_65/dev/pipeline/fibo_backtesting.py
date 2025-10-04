import pandas as pd
import yfinance as yf
from backtesting.fibo_backtesting import StepBacktester
from strategies.fibonacci_strategy import FibonacciStrategy

def backtest():
    # Download historical data
    ticker = "TCS.NS"
    df = yf.download(ticker, period="6mo", interval="1d",multi_level_index=False)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    

    # Initialize strategy
    strategy = FibonacciStrategy(lookback=50)
    

    # Step-by-step backtest
    bt = StepBacktester(df, strategy)
    trades_df = bt.run_step_by_step()
    print(trades_df)
