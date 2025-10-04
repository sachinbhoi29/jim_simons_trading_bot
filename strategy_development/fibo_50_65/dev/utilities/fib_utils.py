# utils/fib_utils.py
import pandas as pd

def last_price_fib_info(df, price_col="Close"):
    """
    Calculate the last price, its percentage between Fib 0% and 100%, 
    and the closest standard Fib level.

    Args:
        df (pd.DataFrame): DataFrame with Fib levels columns.
        price_col (str): Column to use for last price, default "Close".

    Returns:
        dict: {
            "last_price": float,
            "fib_percent": float,
            "closest_fib_level": float
        }
    """
    last_price = df[price_col].iloc[-1]
    fib_levels = df.iloc[-1][["Fib_0%", "Fib_23%", "Fib_38%", "Fib_50%", "Fib_61%", "Fib_78%", "Fib_100%"]]
    
    fib_0 = fib_levels["Fib_0%"]
    fib_100 = fib_levels["Fib_100%"]
    fib_percent = (last_price - fib_0) / (fib_100 - fib_0) * 100
    
    closest_fib_level = fib_levels.iloc[(fib_levels - last_price).abs().argmin()]

    return {
        "last_price": last_price,
        "fib_percent": fib_percent,
        "closest_fib_level": closest_fib_level
    }