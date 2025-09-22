#not validated
import json
import numpy as np
import pandas as pd
from math import isfinite

# Load data
with open('feature_development/options/dev/strategy_output.json') as f:
    strategy_json = json.load(f)

df = pd.read_csv("feature_development/options/dev/NIFTY_options_30Sep2025_with_greeks.csv")

# ---- Helper functions ----
def mid_price(bid, ask, last):
    try:
        if pd.notna(bid) and pd.notna(ask) and isfinite(bid) and isfinite(ask):
            return float((bid + ask) / 2.0)
        if pd.notna(last) and isfinite(last):
            return float(last)
    except Exception:
        pass
    return np.nan

def compute_pop_for_seller_put(row):
    pdlt = row.get('Put_Delta', row.get('PE_delta', np.nan))
    if pd.isna(pdlt):
        return np.nan
    return float(1.0 - abs(pdlt))

def compute_pop_for_seller_call(row):
    cdlt = row.get('Call_Delta', row.get('CE_delta', np.nan))
    if pd.isna(cdlt):
        return np.nan
    return float(1.0 - abs(cdlt))

def safe_get(row, *cols):
    for c in cols:
        if c in row and pd.notna(row[c]):
            return float(row[c])
    return np.nan

# ---- Bull Put Spread ----
def find_best_bull_put_spread(df, max_width=400):
    results = []
    strikes = sorted(df['StrikePrice'].unique())
    strike_rows = {s: df[df['StrikePrice']==s].iloc[0] for s in strikes}

    for i, Ks in enumerate(strikes):
        for Kb in strikes[:i]:
            width = Ks - Kb
            if width <=0 or width > max_width:
                continue

            row_short = strike_rows[Ks]
            row_long = strike_rows[Kb]

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
            if credit <=0 or credit > width:
                continue

            max_profit = float(credit)
            max_loss = float(width - credit)
            risk_reward = float(max_profit / max_loss) if max_loss>0 else None
            pop = compute_pop_for_seller_put(row_short)

            ev = float(pop * max_profit - (1-pop)*max_loss) if pd.notna(pop) else None

            results.append({
                "strategy": "Bull Put Spread",
                "description": f"Sell Put {Ks}, Buy Put {Kb}",
                "short_strike": int(Ks),
                "long_strike": int(Kb),
                "credit": round(credit,2),
                "max_profit": round(max_profit,2),
                "max_loss": round(max_loss,2),
                "risk_reward": round(risk_reward,2) if risk_reward else None,
                "pop": round(pop,2) if pop else None,
                "ev": round(ev,2) if ev else None,
                "break_even": round(Ks - credit,2)
            })
    if not results:
        return None
    # pick best by EV
    best = max(results, key=lambda x: x['ev'] if x['ev'] is not None else -np.inf)
    return best

# ---- Covered Call ----
def find_best_covered_call(df, spot_price):
    results = []
    strikes = sorted(df['StrikePrice'].unique())
    strike_rows = {s: df[df['StrikePrice']==s].iloc[0] for s in strikes}

    for Kc in strikes:
        if Kc < spot_price:
            continue
        row = strike_rows[Kc]
        premium = mid_price(
            safe_get(row,'CE_bidprice','Call_Bid','CE_bidPrice'),
            safe_get(row,'CE_askPrice','Call_Ask','CE_askPrice'),
            safe_get(row,'CE_lastPrice','Call_LTP','CE_lastPrice')
        )
        if pd.isna(premium):
            continue

        max_profit = float((Kc - spot_price) + premium)
        max_loss = float(spot_price - premium)
        break_even = float(spot_price - premium)
        pop = compute_pop_for_seller_call(row)
        ev = float(pop * max_profit - (1-pop) * max_loss) if pd.notna(pop) else None

        results.append({
            "strategy": "Covered Call",
            "description": f"Buy Stock at {round(spot_price,2)}, Sell Call {int(Kc)} at premium {round(premium,2)}",
            "strike": int(Kc),
            "premium": round(premium,2),
            "max_profit": round(max_profit,2),
            "max_loss": round(max_loss,2),
            "break_even": round(break_even,2),
            "pop": round(pop,2) if pop else None
        })

    if not results:
        return None
    best = max(results, key=lambda x: x['max_profit'])
    return best

# ---- Cash-Secured Put ----
def find_best_cash_secured_put(df, spot_price):
    results = []
    strikes = sorted(df['StrikePrice'].unique())
    strike_rows = {s: df[df['StrikePrice']==s].iloc[0] for s in strikes}

    for Kp in strikes:
        if Kp > spot_price:
            continue
        row = strike_rows[Kp]
        premium = mid_price(
            safe_get(row,'PE_bidprice','Put_Bid','PE_bidPrice'),
            safe_get(row,'PE_askPrice','Put_Ask','PE_askPrice'),
            safe_get(row,'PE_lastPrice','Put_LTP','PE_lastPrice')
        )
        if pd.isna(premium):
            continue

        max_profit = float(premium)
        max_loss = float(Kp - premium)
        break_even = float(Kp - premium)
        pop = compute_pop_for_seller_put(row)

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
        return None
    best = max(results, key=lambda x: x['max_profit'])
    return best

# ---- Runner ----
strategy_function_map = {
    'Bull Put Spread (Credit Spread)': find_best_bull_put_spread,
    'Covered Call': find_best_covered_call,
    'Cash-Secured Put': find_best_cash_secured_put
}

def analyze_strategies(df, strategy_json):
    spot = float(strategy_json['market_snapshot']['spot_price'])
    recommendations = {}
    for strat in strategy_json['strategies']:
        name = strat['name']
        func = strategy_function_map.get(name)
        if func is None:
            recommendations[name] = {"note":"No automated scanner implemented"}
            continue
        if name == 'Covered Call':
            best = func(df, spot)
        elif name == 'Cash-Secured Put':
            best = func(df, spot)
        else:
            best = func(df)
        recommendations[name] = best if best else {"note":"No valid leg found"}
    return recommendations

# ---- Run ----
strategy_recommendations = analyze_strategies(df, strategy_json)
with open('feature_development/options/dev/strategy_recommendations.json','w') as f:
    json.dump(strategy_recommendations, f, indent=2)

print(json.dumps(strategy_recommendations, indent=2))
