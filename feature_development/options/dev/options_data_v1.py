from nsepython import nse_optionchain_scrapper
import pandas as pd
import os

def get_options_data(symbol: str, save_dir: str = "feature_development/options/dev") -> str:
    """
    Fetches NSE option chain data for a given symbol and saves it as a CSV.
    
    Args:
        symbol (str): The underlying symbol, e.g., 'NIFTY', 'BANKNIFTY'.
        save_dir (str): Directory where CSV should be saved.
    
    Returns:
        str: Full path to the saved CSV file.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Fetch option chain JSON
    print(f"Fetching option chain data for {symbol}...")
    options_data = nse_optionchain_scrapper(symbol)
    
    # Flatten JSON
    records = options_data['records']['data']
    df = pd.json_normalize(records, sep='_')
    
    # Save CSV
    csv_file = os.path.join(save_dir, f"{symbol}_optionchain_raw.csv")
    df.to_csv(csv_file, index=False)
    
    print(f"Option chain JSON saved to CSV: {csv_file}")
    return csv_file

# ---------------- Example usage ----------------
csv_path = get_options_data("NIFTY")
