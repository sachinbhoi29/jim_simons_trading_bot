import pandas as pd
import numpy as np
from pathlib import Path

# ===============================
# Helper functions
# ===============================
def safe_abs(x):
    return abs(x) if pd.notna(x) else 0.0

def band_score(value, low, high, comfortable_window=0.02):
    """Score a value based on whether it falls in a preferred band."""
    if low <= value <= high:
        return 1.0
    dist = min(abs(value - low), abs(value - high))
    return max(0.0, 1 - dist / (comfortable_window * max(1, abs(value))))

def approximate_margin(option_type, strike, spot_price):
    """
    Rough estimate of margin for a short option (ignores lot size).
    Short Put: margin ~ strike
    Short Call: margin ~ spot_price * 3 (rough cap)
    """
    if option_type == 'P':
        return strike
    elif option_type == 'C':
        return spot_price * 3  # conservative
    return 1e6  # fallback

# ===============================
# Risk-adjusted scoring functions
# ===============================
def score_short_put(row, spot_price):
    strike = row['PE_strikePrice']
    ltp = row['PE_lastPrice']
    delta = safe_abs(row['Put_Delta'])
    theta = -row['Put_Theta_per_day']
    gamma = safe_abs(row['Put_Gamma'])
    vega = safe_abs(row['Put_Vega'])
    dte = row['Days_to_expiry']
    oi = row['PE_openInterest']
    vol = row['PE_totalTradedVolume']

    if pd.isna(ltp) or delta == 0:
        return None

    # 1️⃣ Delta, Theta, Gamma, DTE, Liquidity
    delta_score = band_score(delta, 0.1, 0.3)
    theta_score = np.tanh(max(0, theta)/100)
    gamma_score = -np.tanh(gamma/0.02)
    dte_score = 1.0 if 7 <= dte <= 30 else 0.5
    liq_score = 0.5 * np.log1p(oi + vol)

    # 2️⃣ Risk-adjusted ROI
    margin = approximate_margin('P', strike, spot_price)
    roi_score = min(ltp / margin * 100, 1.0)

    total_score = (
        2.0*delta_score + 1.5*theta_score + 1.0*gamma_score +
        1.0*dte_score + 1.0*liq_score + 3.0*roi_score
    )

    return {"strike": strike, "ltp": ltp, "delta": delta, "theta": theta,
            "gamma": gamma, "vega": vega, "dte": dte, "oi": oi, "vol": vol,
            "score": total_score, "roi_est": ltp/margin}

def score_short_call(row, spot_price):
    strike = row['CE_strikePrice']
    ltp = row['CE_lastPrice']
    delta = safe_abs(row['Call_Delta'])
    theta = -row['Call_Theta_per_day']
    gamma = safe_abs(row['Call_Gamma'])
    vega = safe_abs(row['Call_Vega'])
    dte = row['Days_to_expiry']
    oi = row['CE_openInterest']
    vol = row['CE_totalTradedVolume']

    if pd.isna(ltp) or delta == 0:
        return None

    delta_score = band_score(delta, 0.1, 0.3)
    theta_score = np.tanh(max(0, theta)/100)
    gamma_score = -np.tanh(gamma/0.02)
    dte_score = 1.0 if 7 <= dte <= 30 else 0.5
    liq_score = 0.5 * np.log1p(oi + vol)
    margin = approximate_margin('C', strike, spot_price)
    roi_score = min(ltp / margin * 100, 1.0)

    total_score = (
        2.0*delta_score + 1.5*theta_score + 1.0*gamma_score +
        1.0*dte_score + 1.0*liq_score + 3.0*roi_score
    )

    return {"strike": strike, "ltp": ltp, "delta": delta, "theta": theta,
            "gamma": gamma, "vega": vega, "dte": dte, "oi": oi, "vol": vol,
            "score": total_score, "roi_est": ltp/margin}

def score_long_put(row):
    strike = row['PE_strikePrice']
    ltp = row['PE_lastPrice']
    delta = row['Put_Delta']
    theta = -row['Put_Theta_per_day']
    gamma = safe_abs(row['Put_Gamma'])
    vega = safe_abs(row['Put_Vega'])
    dte = row['Days_to_expiry']
    oi = row['PE_openInterest']
    vol = row['PE_totalTradedVolume']

    if pd.isna(ltp) or delta == 0:
        return None

    delta_score = band_score(abs(delta), 0.5, 0.7)
    theta_score = -np.tanh(theta/100)
    gamma_score = -np.tanh(gamma/0.02)
    dte_score = 1.0 if 7 <= dte <= 45 else 0.5
    liq_score = 0.5 * np.log1p(oi + vol)

    total_score = 2.0*delta_score + 1.0*theta_score + 1.0*gamma_score + 1.0*dte_score + 1.0*liq_score

    return {"strike": strike, "ltp": ltp, "delta": delta, "theta": theta,
            "gamma": gamma, "vega": vega, "dte": dte, "oi": oi, "vol": vol,
            "score": total_score}

def score_long_call(row):
    strike = row['CE_strikePrice']
    ltp = row['CE_lastPrice']
    delta = row['Call_Delta']
    theta = row['Call_Theta_per_day']
    gamma = safe_abs(row['Call_Gamma'])
    vega = safe_abs(row['Call_Vega'])
    dte = row['Days_to_expiry']
    oi = row['CE_openInterest']
    vol = row['CE_totalTradedVolume']

    if pd.isna(ltp) or delta == 0:
        return None

    delta_score = band_score(abs(delta), 0.5, 0.7)
    theta_score = -np.tanh(theta/100)
    gamma_score = -np.tanh(gamma/0.02)
    dte_score = 1.0 if 7 <= dte <= 45 else 0.5
    liq_score = 0.5 * np.log1p(oi + vol)

    total_score = 2.0*delta_score + 1.0*theta_score + 1.0*gamma_score + 1.0*dte_score + 1.0*liq_score

    return {"strike": strike, "ltp": ltp, "delta": delta, "theta": theta,
            "gamma": gamma, "vega": vega, "dte": dte, "oi": oi, "vol": vol,
            "score": total_score}

# ===============================
# Generic risk-adjusted optimizer
# ===============================
def optimize_leg(df, option_type='C', position='Buy', top_n=10, spot_price=1):
    scoring_map = {
        ('C','Buy'): score_long_call,
        ('C','Sell'): score_short_call,
        ('P','Buy'): score_long_put,
        ('P','Sell'): score_short_put
    }

    key = (option_type.upper(), position.capitalize())
    if key not in scoring_map:
        raise NotImplementedError(f"Leg type {key} not implemented")

    scorer = scoring_map[key]
    candidates = []

    for _, row in df.iterrows():
        if position.lower() == 'sell':
            scored = scorer(row, spot_price)
        else:
            scored = scorer(row)
        if scored is not None:
            candidates.append(scored)

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_n]

# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    csv_path = Path("feature_development/options/dev/BANKNIFTY_options_30Sep2025_with_greeks.csv")
    df = pd.read_csv(csv_path)

    spot_price = df['PE_underlyingValue'].dropna().iloc[0]

    # Short puts
    short_puts = optimize_leg(df, option_type='P', position='Sell', top_n=10, spot_price=spot_price)
    print("Optimized Short Puts:\n", pd.DataFrame(short_puts))

    # Short calls
    short_calls = optimize_leg(df, option_type='C', position='Sell', top_n=10, spot_price=spot_price)
    print("Optimized Short Calls:\n", pd.DataFrame(short_calls))
