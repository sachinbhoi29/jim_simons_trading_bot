import pandas as pd
import numpy as np
from pathlib import Path

# ===============================
# Helper functions
# ===============================
def get_spot(df):
    return df['PE_underlyingValue'].dropna().iloc[0]

def get_atm_strike(df):
    spot = get_spot(df)
    all_strikes = sorted(df['PE_strikePrice'].dropna().unique())
    return min(all_strikes, key=lambda x: abs(x - spot))

def weekly_move(spot, iv, days=7):
    """Expected weekly move in points"""
    return spot * iv * np.sqrt(days/252)

def generate_weekly_ic(df, wing_distance_factor=1.5):
    """
    Generate weekly Iron Condors based on expected weekly move
    wing_distance_factor: how far wings are from short strikes
    """
    spot = get_spot(df)
    atm = get_atm_strike(df)
    atm_iv = df.loc[df['PE_strikePrice']==atm, 'PE_impliedVolatility'].values[0] / 100

    move = weekly_move(spot, atm_iv, days=7)
    move = round(move / 100) * 100  # round to nearest 100 points

    # Short strikes
    sp_short = atm - move
    sc_short = atm + move

    # Long strikes (wings)
    width = move * wing_distance_factor
    lp_long = sp_short - width
    lc_long = sc_short + width

    # Make sure strikes exist in chain
    all_puts = sorted(df['PE_strikePrice'].dropna().unique())
    all_calls = sorted(df['CE_strikePrice'].dropna().unique())

    sp_short = min(all_puts, key=lambda x: abs(x - sp_short))
    lp_long = min(all_puts, key=lambda x: abs(x - lp_long))
    sc_short = min(all_calls, key=lambda x: abs(x - sc_short))
    lc_long = min(all_calls, key=lambda x: abs(x - lc_long))

    ic = {'short_put': sp_short, 'long_put': lp_long, 'short_call': sc_short, 'long_call': lc_long}
    return [ic]

def rank_ic_weekly(df, ics):
    ranked = []
    spot = get_spot(df)

    for ic in ics:
        sp_price = df.loc[df['PE_strikePrice']==ic['short_put'], 'PE_lastPrice'].values[0]
        lp_price = df.loc[df['PE_strikePrice']==ic['long_put'], 'PE_lastPrice'].values[0]
        sc_price = df.loc[df['CE_strikePrice']==ic['short_call'], 'CE_lastPrice'].values[0]
        lc_price = df.loc[df['CE_strikePrice']==ic['long_call'], 'CE_lastPrice'].values[0]

        credit = sp_price + sc_price
        width_put = ic['short_put'] - ic['long_put']
        width_call = ic['long_call'] - ic['short_call']
        max_width = max(width_put, width_call)

        roi = credit / max_width if max_width != 0 else 0

        ic_full = {
            'ROI': roi,
            'Credit': credit,
            'MaxLoss': max_width,
            'Legs': [
                {'Type': 'Put', 'Action': 'Sell', 'Strike': ic['short_put'], 'LTP': sp_price},
                {'Type': 'Put', 'Action': 'Buy',  'Strike': ic['long_put'],  'LTP': lp_price},
                {'Type': 'Call','Action': 'Sell', 'Strike': ic['short_call'],'LTP': sc_price},
                {'Type': 'Call','Action': 'Buy',  'Strike': ic['long_call'], 'LTP': lc_price},
            ]
        }
        ranked.append(ic_full)

    ranked.sort(key=lambda x: x['ROI'], reverse=True)
    return ranked

# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    csv_path = Path("feature_development/options/dev/BANKNIFTY_options_30Sep2025_with_greeks.csv")
    df = pd.read_csv(csv_path)

    # Step 1: Generate weekly IC
    ics = generate_weekly_ic(df, wing_distance_factor=1.5)

    # Step 2: Rank IC
    ranked_ics = rank_ic_weekly(df, ics)

    # Step 3: Print
    for i, ic in enumerate(ranked_ics, start=1):
        print(f"\nWeekly IC #{i}")
        print(f"ROI: {ic['ROI']:.2f}, Credit: {ic['Credit']:.2f}, Max Loss: {ic['MaxLoss']:.2f}")
        for leg in ic['Legs']:
            print(f"  {leg['Action']} {leg['Type']}: Strike={leg['Strike']}, LTP={leg['LTP']}")
