import pandas as pd
import numpy as np
import json
from pathlib import Path

# ===============================
# Helper functions
# ===============================
def safe_abs(x):
    return abs(x) if pd.notna(x) else 0.0

def band_score(value, low, high, comfortable_window=0.02):
    """
    Score 1 if within target band, linearly decays outside.
    """
    if low <= value <= high:
        return 1.0
    dist = min(abs(value - low), abs(value - high))
    return max(0.0, 1 - dist / (comfortable_window * max(1, abs(value))))

# ===============================
# Leg scoring functions
# ===============================
def score_long_call(row):
    delta = row['Call_Delta']
    theta = row['Call_Theta_per_day']
    gamma = safe_abs(row['Call_Gamma'])
    vega  = safe_abs(row['Call_Vega'])
    ltp   = row['CE_lastPrice']
    dte   = row['Days_to_expiry']
    oi    = row['CE_openInterest']
    vol   = row['CE_totalTradedVolume']

    if pd.isna(ltp) or delta == 0:
        return None

    score = 0
    score += 1.5 * band_score(delta, 0.5, 0.7)
    score += 0.5 * np.tanh(theta/100)
    score -= 1.0 * np.tanh(gamma/0.02)
    score += 0.5 * (1.0 if 7 <= dte <= 45 else 0.5)
    score += 0.5 * np.log1p(oi + vol)

    return {"strike": row['CE_strikePrice'], "ltp": ltp, "delta": delta,
            "theta": theta, "gamma": gamma, "vega": vega, "dte": dte,
            "oi": oi, "vol": vol, "score": score}

def score_short_call(row):
    delta = safe_abs(row['Call_Delta'])
    theta = -row['Call_Theta_per_day']
    gamma = safe_abs(row['Call_Gamma'])
    vega  = safe_abs(row['Call_Vega'])
    ltp   = row['CE_lastPrice']
    dte   = row['Days_to_expiry']
    oi    = row['CE_openInterest']
    vol   = row['CE_totalTradedVolume']

    if pd.isna(ltp) or delta == 0:
        return None

    score = 0
    score += 1.5 * band_score(delta, 0.1, 0.3)
    score += 2.0 * np.tanh(max(0, theta)/100)
    score -= 1.0 * np.tanh(gamma/0.02)
    score += 0.5 * (1.0 if 7 <= dte <= 45 else 0.5)
    score += 0.5 * np.log1p(oi + vol)

    return {"strike": row['CE_strikePrice'], "ltp": ltp, "delta": delta,
            "theta": theta, "gamma": gamma, "vega": vega, "dte": dte,
            "oi": oi, "vol": vol, "score": score}

def score_long_put(row):
    delta = row['Put_Delta']
    theta = row['Put_Theta_per_day']
    gamma = safe_abs(row['Put_Gamma'])
    vega  = safe_abs(row['Put_Vega'])
    ltp   = row['PE_lastPrice']
    dte   = row['Days_to_expiry']
    oi    = row['PE_openInterest']
    vol   = row['PE_totalTradedVolume']

    if pd.isna(ltp) or delta == 0:
        return None

    score = 0
    score += 1.5 * band_score(delta, 0.5, 0.7)
    score += 0.5 * np.tanh(theta/100)
    score -= 1.0 * np.tanh(gamma/0.02)
    score += 0.5 * (1.0 if 7 <= dte <= 45 else 0.5)
    score += 0.5 * np.log1p(oi + vol)

    return {"strike": row['PE_strikePrice'], "ltp": ltp, "delta": delta,
            "theta": theta, "gamma": gamma, "vega": vega, "dte": dte,
            "oi": oi, "vol": vol, "score": score}

def score_short_put(row):
    delta = safe_abs(row['Put_Delta'])
    theta = -row['Put_Theta_per_day']
    gamma = safe_abs(row['Put_Gamma'])
    vega  = safe_abs(row['Put_Vega'])
    ltp   = row['PE_lastPrice']
    dte   = row['Days_to_expiry']
    oi    = row['PE_openInterest']
    vol   = row['PE_totalTradedVolume']

    if pd.isna(ltp) or delta == 0:
        return None

    score = 0
    score += 1.5 * band_score(delta, 0.1, 0.3)
    score += 2.0 * np.tanh(max(0, theta)/100)
    score -= 1.0 * np.tanh(gamma/0.02)
    score += 0.5 * (1.0 if 7 <= dte <= 45 else 0.5)
    score += 0.5 * np.log1p(oi + vol)

    return {"strike": row['PE_strikePrice'], "ltp": ltp, "delta": delta,
            "theta": theta, "gamma": gamma, "vega": vega, "dte": dte,
            "oi": oi, "vol": vol, "score": score}

# ===============================
# Generic leg optimizer
# ===============================
def optimize_leg(df, option_type='C', position='Buy', top_n=5):
    """
    option_type: 'C' = Call, 'P' = Put
    position: 'Buy' = Long, 'Sell' = Short
    """
    scoring_map = {
        ('C','Buy'): score_long_call,
        ('C','Sell'): score_short_call,
        ('P','Buy'): score_long_put,
        ('P','Sell'): score_short_put
    }

    key = (option_type.upper(), position.capitalize())
    if key not in scoring_map:
        raise NotImplementedError(f"Leg type {key} not implemented.")

    scorer = scoring_map[key]
    candidates = [scorer(row) for _, row in df.iterrows() if scorer(row) is not None]
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_n]

# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    csv_path = Path("feature_development/options/dev/BANKNIFTY_options_30Sep2025_with_greeks.csv")
    df = pd.read_csv(csv_path)

    leg = optimize_leg(df, option_type='P', position='Sell')  # Short Put
    print("Top Short Put candidates:\n", pd.DataFrame(leg))
