import pandas as pd
import numpy as np
import yfinance as yf

# --- Step 1: Fetch Realized Volatility ---
def get_realized_vol(symbol: str, window: int = 30):
    ticker_map = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK"
    }
    
    if symbol not in ticker_map:
        raise ValueError(f"Symbol {symbol} not in ticker_map")

    data = yf.download(ticker_map[symbol], period="1y", interval="1d")
    data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data['rv'] = data['log_ret'].rolling(window).std() * np.sqrt(252)
    
    return data[['Close', 'rv']].dropna()

# --- Step 2: Compare IV with Realized Vol ---
def compare_iv_rv(options_df: pd.DataFrame, window: int = 30):
    results = []
    for symbol in options_df['CE_underlying'].unique():
        rv_data = get_realized_vol(symbol, window=window)
        latest_rv = rv_data['rv'].iloc[-1]

        subset = options_df[options_df['CE_underlying'] == symbol].copy()
        subset['RealizedVol'] = latest_rv
        
        # Convert IV from percent to decimal
        subset['Call_IV_dec'] = subset['Call_IV'] / 100
        subset['Put_IV_dec'] = subset['Put_IV'] / 100

        # IV-RV spread for call and put
        subset['Call_IV-RV_Spread'] = subset['Call_IV_dec'] - latest_rv
        subset['Put_IV-RV_Spread'] = subset['Put_IV_dec'] - latest_rv

        results.append(subset)
    
    return pd.concat(results, ignore_index=True)

# --- Step 3: Rank Strikes and Generate Signal (dynamic range) ---
def rank_strikes_with_signals(df: pd.DataFrame, below_pct=0.05, above_pct=0.05):
    # Compute dynamic strike range based on spot price
    df['LowerStrike'] = df['CE_underlyingValue'] * (1 - below_pct)
    df['UpperStrike'] = df['CE_underlyingValue'] * (1 + above_pct)
    
    # Filter strikes within dynamic range
    df = df[(df['StrikePrice'] >= df['LowerStrike']) & 
            (df['StrikePrice'] <= df['UpperStrike'])].copy()
    
    # Absolute imbalance
    df['AbsImbalance_Call'] = df['Call_IV-RV_Spread'].abs()
    df['AbsImbalance_Put'] = df['Put_IV-RV_Spread'].abs()
    df['Max_AbsImbalance'] = df[['AbsImbalance_Call', 'AbsImbalance_Put']].max(axis=1)
    
    # Signal: which option is overpriced
    def get_signal(row):
        if row['AbsImbalance_Call'] > row['AbsImbalance_Put']:
            return "Sell Call" if row['Call_IV-RV_Spread'] > 0 else "Buy Call"
        else:
            return "Sell Put" if row['Put_IV-RV_Spread'] > 0 else "Buy Put"
    
    df['Signal'] = df.apply(get_signal, axis=1)
    
    # Rank by max imbalance
    df = df.sort_values(by='Max_AbsImbalance', ascending=False)
    df['Rank'] = np.arange(1, len(df)+1)
    
    return df[[
        "Rank",
        "CE_underlying",
        "StrikePrice",
        "Call_IV",
        "Put_IV",
        "RealizedVol",
        "Call_IV-RV_Spread",
        "Put_IV-RV_Spread",
        "Max_AbsImbalance",
        "Signal"
    ]]

# --- Step 4: Load CSV and Apply Functions ---
options_df = pd.read_csv("feature_development/options/dev/BANKNIFTY_options_30Sep2025_with_greeks.csv")

comparison = compare_iv_rv(options_df, window=30)
ranked_strikes = rank_strikes_with_signals(comparison, below_pct=0.05, above_pct=0.05)

# --- Step 5: Save Final CSV ---
output_path = "feature_development/options/dev/compare_iv_ranked_call_put_dynamic.csv"
ranked_strikes.to_csv(output_path, index=False)

print(f"Full ranked comparison saved to {output_path}")
print(ranked_strikes.head(20))
