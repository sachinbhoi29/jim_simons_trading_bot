import json
import numpy as np
import pandas as pd
from math import isfinite

# ---- Helper functions ----
def mid_price(bid, ask, last):
    """Compute mid price; fallback to last price."""
    try:
        if bid and ask and isfinite(bid) and isfinite(ask):
            return float((bid + ask) / 2.0)
        if last and isfinite(last):
            return float(last)
    except Exception:
        pass
    return np.nan

def safe_get(row, *cols):
    """Return first valid numeric value from row for given columns."""
    for c in cols:
        if c in row and pd.notna(row[c]) and row[c] != 0:
            return float(row[c])
    return np.nan

def compute_pop_put(row):
    """Probability of profit for short put (seller)."""
    delta = safe_get(row, 'Put_Delta', 'PE_delta')
    if pd.isna(delta):
        return np.nan
    return 1 - abs(delta)

def compute_pop_call(row):
    """Probability of profit for short call (seller)."""
    delta = safe_get(row, 'Call_Delta', 'CE_delta')
    if pd.isna(delta):
        return np.nan
    return 1 - abs(delta)

# ---- Bull Put Spread ----
def find_best_bull_put_spread(df, spot, max_width=400, max_dist=500):
    results = []
    strikes = sorted(df['StrikePrice'].unique())
    strike_rows = {s: df[df['StrikePrice']==s].iloc[0] for s in strikes}

    for i, Ks in enumerate(strikes):
        for Kb in strikes[:i]:
            width = Ks - Kb
            # skip if width invalid or strikes too far from spot
            if width <= 0 or width > max_width or abs(Ks - spot) > max_dist or abs(Kb - spot) > max_dist:
                continue

            row_short, row_long = strike_rows[Ks], strike_rows[Kb]

            short_premium = mid_price(
                safe_get(row_short,'PE_bidprice','Put_Bid','PE_bidPrice'),
                safe_get(row_short,'PE_askPrice','Put_Ask','PE_ask'),
                safe_get(row_short,'PE_lastPrice','Put_LTP','PE_lastPrice')
            )
            long_premium = mid_price(
                safe_get(row_long,'PE_bidprice','Put_Bid','PE_bidPrice'),
                safe_get(row_long,'PE_askPrice','Put_Ask','PE_ask'),
                safe_get(row_long,'PE_lastPrice','Put_LTP','PE_lastPrice')
            )
            if pd.isna(short_premium) or pd.isna(long_premium):
                continue

            credit = short_premium - long_premium
            if credit <= 0 or credit > width:
                continue

            max_profit = credit
            max_loss = width - credit
            pop = compute_pop_put(row_short)
            ev = pop * max_profit - (1-pop) * max_loss if pd.notna(pop) else None

            results.append({
                "strategy": "Bull Put Spread",
                "description": f"Sell Put {Ks}, Buy Put {Kb}",
                "short_strike": int(Ks),
                "long_strike": int(Kb),
                "credit": round(credit,2),
                "max_profit": round(max_profit,2),
                "max_loss": round(max_loss,2),
                "risk_reward": round(max_profit/max_loss,2) if max_loss>0 else None,
                "pop": round(pop,2) if pop else None,
                "ev": round(ev,2) if ev else None,
                "break_even": round(Ks - credit,2)
            })

    if not results:
        return {"note":"No valid Bull Put Spread found near spot"}
    return max(results, key=lambda x: x['ev'] if x['ev'] is not None else -np.inf)

# ---- Covered Call ----
def find_best_covered_call(df, spot, max_dist=500):
    results = []
    strikes = sorted(df['StrikePrice'].unique())
    strike_rows = {s: df[df['StrikePrice']==s].iloc[0] for s in strikes}

    for Kc in strikes:
        if Kc < spot or abs(Kc - spot) > max_dist:
            continue

        row = strike_rows[Kc]
        premium = mid_price(
            safe_get(row,'CE_bidprice','Call_Bid','CE_bidPrice'),
            safe_get(row,'CE_askPrice','Call_Ask','CE_askPrice'),
            safe_get(row,'CE_lastPrice','Call_LTP','CE_lastPrice')
        )
        if pd.isna(premium):
            continue

        max_profit = (Kc - spot) + premium
        max_loss = spot - premium
        break_even = spot - premium
        pop = compute_pop_call(row)
        ev = pop * max_profit - (1-pop) * max_loss if pd.notna(pop) else None

        results.append({
            "strategy": "Covered Call",
            "description": f"Buy Stock at {round(spot,2)}, Sell Call {int(Kc)} at premium {round(premium,2)}",
            "strike": int(Kc),
            "premium": round(premium,2),
            "max_profit": round(max_profit,2),
            "max_loss": round(max_loss,2),
            "break_even": round(break_even,2),
            "pop": round(pop,2) if pop else None
        })

    if not results:
        return {"note":"No valid Covered Call found near spot"}
    return max(results, key=lambda x: x['max_profit'])

# ---- Cash-Secured Put ----
def find_best_cash_secured_put(df, spot, max_dist=500):
    results = []
    strikes = sorted(df['StrikePrice'].unique())
    strike_rows = {s: df[df['StrikePrice']==s].iloc[0] for s in strikes}

    for Kp in strikes:
        if Kp > spot or abs(Kp - spot) > max_dist:
            continue

        row = strike_rows[Kp]
        premium = mid_price(
            safe_get(row,'PE_bidprice','Put_Bid','PE_bidPrice'),
            safe_get(row,'PE_askPrice','Put_Ask','PE_askPrice'),
            safe_get(row,'PE_lastPrice','Put_LTP','PE_lastPrice')
        )
        if pd.isna(premium):
            continue

        max_profit = premium
        max_loss = Kp - premium
        break_even = Kp - premium
        pop = compute_pop_put(row)

        results.append({
            "strategy": "Cash-Secured Put",
            "description": f"Sell Put {int(Kp)} (cash-secured) at premium {round(premium,2)}",
            "strike": int(Kp),
            "premium": round(premium,2),
            "max_profit": round(max_profit,2),
            "max_loss": round(max_loss,2),
            "break_even": round(break_even,2),
            "pop": round(pop,2) if pop else None
        })

    if not results:
        return {"note":"No valid Cash-Secured Put found near spot"}
    return max(results, key=lambda x: x['max_profit'])

# ---- Runner ----
strategy_function_map = {
    'Bull Put Spread (Credit Spread)': find_best_bull_put_spread,
    'Covered Call': find_best_covered_call,
    'Cash-Secured Put': find_best_cash_secured_put
}

def analyze_strategies(df, strategy_json, max_dist=500):
    spot = float(strategy_json['market_snapshot']['spot_price'])
    recommendations = {}
    for strat in strategy_json['strategies']:
        name = strat['name']
        func = strategy_function_map.get(name)
        if func is None:
            recommendations[name] = {"note":"No automated scanner implemented"}
            continue
        if name == 'Covered Call':
            best = func(df, spot, max_dist)
        elif name == 'Cash-Secured Put':
            best = func(df, spot, max_dist)
        else:
            best = func(df, spot, max_dist)
        recommendations[name] = best
    return recommendations

# ---- Run ----
df = pd.read_csv("feature_development/options/dev/NIFTY_options_30Sep2025_with_greeks.csv")
with open('feature_development/options/dev/strategy_output.json') as f:
    strategy_json = json.load(f)

strategy_recommendations = analyze_strategies(df, strategy_json, max_dist=500)

with open('feature_development/options/dev/strategy_recommendations.json','w') as f:
    json.dump(strategy_recommendations, f, indent=2)

print(json.dumps(strategy_recommendations, indent=2))
