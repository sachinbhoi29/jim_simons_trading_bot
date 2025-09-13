import pandas as pd

def merge_stock_with_nifty_regime(stock_df: pd.DataFrame, nifty_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges stock-level dataframe with NIFTY50 regime dataframe on date,
    standardizes date formats, and prefixes NIFTY columns.
    
    Parameters:
        stock_df (pd.DataFrame): Stock-level dataframe (expects 'Date' as DD-MMM-YY string)
        nifty_df (pd.DataFrame): NIFTY index dataframe with regime, indicators, forecast

    Returns:
        pd.DataFrame: Merged dataframe with nifty50_ columns added
    """
    # Convert stock date to datetime
    stock_df = stock_df.copy()
    # stock_df["Date"] = pd.to_datetime(stock_df["Date"], format="%d-%b-%y")
    
    stock_df["Date"] = pd.to_datetime(stock_df["Date"], format="mixed", dayfirst=True)

    
    # Convert NIFTY date to naive datetime (drop timezone)
    nifty_df = nifty_df.copy()
    nifty_df = nifty_df.reset_index()
    nifty_df["Date"] = pd.to_datetime(nifty_df["Date"]).dt.tz_localize(None)

    # Select useful NIFTY columns
    nifty_cols = [
        "Date", "Close", "50EMA", "200EMA", "RSI", "ATR", "MACD", "Signal",
        "BB_Width", "OBV", "Support", "Resistance", "Enhanced_Regime", "Forecasted Regime"
    ]
    nifty_df = nifty_df[nifty_cols]

    # Rename with prefix
    nifty_df = nifty_df.rename(columns={col: f"nifty50_{col}" for col in nifty_df.columns if col != "Date"})

    # Merge on Date
    # you know what, the nifty50 forecast is getting chopped off because of the left join on the stock df, so using the full join
    # merged_df = pd.merge(stock_df, nifty_df, how="left", left_on="Date", right_on="Date")
    merged_df = pd.merge(stock_df, nifty_df, how="outer", on="Date").sort_values("Date").reset_index(drop=True)


    return merged_df

def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames alternate column names to standard OHLCV format.

    Standardized columns:
        - 'Open'
        - 'High'
        - 'Low'
        - 'Close'
        - 'Volume'

    Accepts typical NSE/BSE column names like 'OpenPrice', 'ClosePrice', 'TotalTradedQuantity', etc.

    Args:
        df (pd.DataFrame): Raw DataFrame with market data.

    Returns:
        pd.DataFrame: Renamed DataFrame with standardized OHLCV columns.
    """
    col_map = {
        "OpenPrice": "Open",
        "HighPrice": "High",
        "LowPrice": "Low",
        "ClosePrice": "Close",
        "TotalTradedQuantity": "Volume",
        "TradedQty": "Volume", 
    }

    # Only rename columns that exist in the DataFrame
    renamed_df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Convert all relevant columns to numeric (after removing commas)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in renamed_df.columns:
            renamed_df[col] = renamed_df[col].astype(str).str.replace(",", "").astype(float)

    return renamed_df
