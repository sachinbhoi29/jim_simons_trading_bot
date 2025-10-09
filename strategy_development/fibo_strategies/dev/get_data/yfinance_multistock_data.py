import yfinance as yf

def download_and_split(
    tickers, 
    start=None, 
    end=None, 
    period=None, 
    interval="1d", 
    auto_adjust=False
):
    """
    Download OHLCV data for multiple tickers and split into individual DataFrames.

    Args:
        tickers (list or str): List of tickers or a single ticker symbol.
        start (str): Start date (YYYY-MM-DD). Used only if period is None.
        end (str): End date (YYYY-MM-DD). Used only if period is None.
        period (str): yfinance period string, e.g. '1y', '6mo', '2y'.
        interval (str): Data interval, e.g. '1d', '1wk', '1h'.
        auto_adjust (bool): Adjust prices for splits/dividends.

    Returns:
        dict: {ticker: DataFrame}
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        multi_level_index=False
    )

    # Put ticker as first column level
    df = df.swaplevel(axis=1).sort_index(axis=1)

    # Split into one DataFrame per ticker
    dfs = {ticker: df[ticker].copy() for ticker in df.columns.get_level_values(0).unique()}

    return dfs

if __name__ == '__main__':
    # Example 1: using start & end
    dfs = download_and_split(["AAPL", "MSFT"], start="2024-01-01", end="2025-04-08")
    print(dfs["AAPL"].head())

    # Example 2: using period
    dfs = download_and_split(["AAPL", "MSFT"], period="1y")
    print(dfs["MSFT"].head())
