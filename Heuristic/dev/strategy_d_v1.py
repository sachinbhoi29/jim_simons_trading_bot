import os
import math
import numpy as np
import pandas as pd

def apply_strategy_d_momentum_pullback(
        csv_path,
        out_signals_path=r"C:\PERSONAL_DATA\Startups\Stocks\Jim_Simons_Trading_Strategy\Heuristic\dev\data\signals_only.csv",
        require_index_bullish=True,
        pullback_atr_mult=1.0,   # allow pullback to EMA20 - 1*ATR
        pullback_pct=0.03,       # or within 3% above/below EMA20
        min_volume_mult=0.8,     # allow slightly lower than avg vol during pullback
        rsi_min=35,              # RSI lower bound for healthy pullback zone
        rsi_max=55):             # RSI upper bound (should not be overbought on entry)

    # --------------------
    # Load data
    # --------------------
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.sort_values("Date", inplace=True)
    df.set_index('Date', inplace=True, drop=False)

    required_cols = [
        "Close","High","Low","Open","Volume","ATR_14","EMA_20","EMA_50","EMA_200",
        "RSI_14","MACD","MACD_signal","%K","%D","VWAP","NIFTY_Trend"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --------------------
    # Derived helpers
    # --------------------
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["ema20_minus_atr"] = df["EMA_20"] - (df["ATR_14"] * pullback_atr_mult)

    # Trend filter: clear bullish alignment
    df["trend_ok"] = (df["EMA_20"] > df["EMA_50"]) & (df["EMA_50"] > df["EMA_200"])

    # Pullback condition (either touch EMA20-ATR or within pullback_pct of EMA20)
    df["pullback_to_atr"] = df["Low"] <= df["ema20_minus_atr"]
    df["pullback_pct_ok"] = ((df["Close"] / df["EMA_20"]) >= (1 - pullback_pct)) & ((df["Close"] / df["EMA_20"]) <= (1 + 0.02))
    # Allow either: deep ATR touch OR mild pullback into EMA zone
    df["pullback_ok"] = df["pullback_to_atr"] | df["pullback_pct_ok"]

    # Momentum re-acceleration (must show signs of turn-up)
    # Primary: MACD above signal or MACD rising
    df["macd_cross_ok"] = df["MACD"] > df["MACD_signal"]
    df["macd_rising"] = df["MACD"] > df["MACD"].shift(1)

    # Secondary: RSI in healthy pullback zone and turning up
    df["rsi_zone_ok"] = df["RSI_14"].between(rsi_min, rsi_max)
    df["rsi_turn_up"] = df["RSI_14"] > df["RSI_14"].shift(1)

    # Stochastic confirming momentum
    df["stoch_ok"] = df["%K"] > df["%D"]

    # Combine momentum signals: require MACD cross OR (RSI zone + (rsi_turn_up OR stoch_ok))
    df["momentum_ok"] = df["macd_cross_ok"] | (df["rsi_zone_ok"] & (df["rsi_turn_up"] | df["stoch_ok"]))

    # Volume: allow slightly lower volume during pullback, but not zero
    df["volume_ok"] = df["Volume"] > df["vol_ma20"] * min_volume_mult
    # Also prefer volume pickup on confirmation bar: today's volume >= yesterday's
    df["volume_pickup"] = df["Volume"] >= df["Volume"].shift(1)

    # Avoid overextended entries: price not too far above EMA20
    df["not_overextended"] = df["Close"] < df["EMA_20"] * 1.04

    # Index regime filter
    df["index_ok"] = True
    if require_index_bullish:
        df["index_ok"] = df["NIFTY_Trend"].str.contains("Bullish", case=False, na=False)

    # Final composite signal:
    # - trend OK
    # - pullback OK
    # - momentum OK
    # - volume OK (and preferably pickup)  <-- pickup not strictly required
    # - not overextended
    # - index OK
    df["take_trade"] = (
        df["trend_ok"] &
        df["pullback_ok"] &
        df["momentum_ok"] &
        df["volume_ok"] &
        df["not_overextended"] &
        df["index_ok"]
    )

    # Optional: require volume pickup as well (uncomment to make stricter)
    # df["take_trade"] = df["take_trade"] & df["volume_pickup"]

    # Extract signals and save
    signals = df[df["take_trade"]].copy()
    signals = signals.reset_index(drop=True)
    signals.to_csv(out_signals_path, index=False)

    print(f"\nSaved {len(signals)} signals → {out_signals_path}")
    return signals


if __name__ == "__main__":
    csv_path = r"C:\PERSONAL_DATA\Startups\Stocks\Jim_Simons_Trading_Strategy\Heuristic\dev\data\all_stocks_full_suite_v2.csv"
    signals = apply_strategy_d_momentum_pullback(csv_path)
