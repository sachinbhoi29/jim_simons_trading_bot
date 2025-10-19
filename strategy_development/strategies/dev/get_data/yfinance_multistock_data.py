import yfinance as yf
import pandas as pd

from pandas.tseries.offsets import MonthEnd

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



def download_and_split_index(
    tickers, 
    start=None, 
    end=None, 
    period=None, 
    interval="1d", 
    auto_adjust=False
):
    """
    Download OHLCV data for multiple tickers and ensure full coverage of requested start-end period.
    Fills missing months by downloading month-by-month.

    Args:
        tickers (list or str): List of tickers or a single ticker symbol.
        start (str): Start date in 'YYYY-MM-DD'. Used only if period is None.
        end (str): End date in 'YYYY-MM-DD'. Used only if period is None.
        period (str): yfinance period string, e.g. '1y', '6mo', '2y'.
        interval (str): Data interval, e.g. '1d', '1wk', '1h'.
        auto_adjust (bool): Adjust prices for splits/dividends.

    Returns:
        dict: {ticker: DataFrame}
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    dfs = {}

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        # Try normal download first
        df = yf.download(
            ticker,
            start=start,
            end=end,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            group_by='ticker',
            threads=True
        )

        # Ensure datetime index and sorted
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

        # --- Fill missing months if initial download does not cover full start date ---
        if start is not None:
            requested_start = pd.to_datetime(start)
            if df.empty or df.index[0] > requested_start:
                print(f"{ticker}: filling missing data from {requested_start.date()} to {df.index[0].date() if not df.empty else pd.to_datetime(end).date()}")

                missing_chunks = []
                current_start = requested_start
                current_end = df.index[0] - pd.Timedelta(days=1) if not df.empty else pd.to_datetime(end)

                while current_start <= current_end:
                    chunk_end = min(current_start + MonthEnd(1) - pd.Timedelta(days=1), current_end)
                    print(f"Downloading missing chunk: {current_start.date()} to {chunk_end.date()}")
                    chunk = yf.download(
                        ticker,
                        start=current_start.date(),
                        end=(chunk_end + pd.Timedelta(days=1)).date(),
                        interval=interval,
                        auto_adjust=auto_adjust
                    )
                    if not chunk.empty:
                        missing_chunks.append(chunk)
                    current_start = chunk_end + pd.Timedelta(days=1)

                if missing_chunks:
                    df_missing = pd.concat(missing_chunks)
                    df_missing.index = pd.to_datetime(df_missing.index)
                    df_missing.sort_index(inplace=True)
                    # Combine missing + existing
                    df = pd.concat([df_missing, df])
                    df = df[~df.index.duplicated(keep='first')]

        # Flatten columns if multi-level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)

        # Ensure required columns exist
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = pd.NA

        dfs[ticker] = df
    return dfs




if __name__ == '__main__':
    # Example 1: using start & end
    dfs = download_and_split(["AAPL", "MSFT"], start="2024-01-01", end="2025-04-08")
    print(dfs["AAPL"].head())

    # Example 2: using period
    dfs = download_and_split(["AAPL", "MSFT"], period="1y")
    print(dfs["MSFT"].head())









