import os
import numpy as np
import pandas as pd

def apply_strategy_c_mean_reversion_relaxed(
        csv_path,
        out_signals_path=r"C:\PERSONAL_DATA\Startups\Stocks\Jim_Simons_Trading_Strategy\Heuristic\dev\data\signals_only.csv",
        require_index_bullish=True):

    # Load data
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True, drop=False)

    required_cols = [
        "Close","High","Low","Volume","ATR_14","EMA_20","EMA_50","EMA_200",
        "RSI_14","MACD","MACD_signal","%K","%D","VWAP","NIFTY_Trend",
        "BB_upper_20","BB_lower_20","BB_mid_20"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ============================================================
    #   STRATEGY C — MEAN REVERSION RELAXED VERSION
    # ============================================================

    # 1. Trend (relaxed) — only avoid downtrends
    df["htf_trend_ok"] = df["EMA_20"] > df["EMA_50"]

    # 2. Oversold (relaxed thresholds)
    df["oversold_rsi"] = df["RSI_14"] < 45      # was 38
    df["oversold_stoch"] = df["%K"] < 40        # was <25

    # 3. Volatility flush (relaxed)
    df["atr_spike"] = df["ATR_14"] > df["ATR_14"].rolling(14).mean() * 1.05
    df["vix_calm"] = df["ATR_14"] < df["ATR_14"].rolling(5).mean() * 1.15

    df["volatility_ok"] = df["atr_spike"] & df["vix_calm"]

    # 4. Selling exhaustion (relaxed)
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["low_volume_pullback"] = df["Volume"] < df["vol_ma20"] * 1.0   # earlier was *0.8

    # 5. Reversal confirmation (relaxed)
    df["rsi_turn_up"] = df["RSI_14"] > df["RSI_14"].shift(1)
    df["stoch_turn_up"] = df["%K"] > df["%D"]

    df["reversal_ok"] = df["rsi_turn_up"] | df["stoch_turn_up"]

    # 6. Price confirmation (relaxed)
    df["close_recovery_ok"] = df["Close"] > df["Low"].rolling(3).min() * 1.03

    # Index filter
    df["index_ok"] = True
    if require_index_bullish:
        df["index_ok"] = df["NIFTY_Trend"].str.contains("Bullish", case=False, na=False)

    df["take_trade"] = (
        df["htf_trend_ok"] &
        df["oversold_rsi"] &
        df["oversold_stoch"] &
        df["volatility_ok"] &
        df["low_volume_pullback"] &
        df["reversal_ok"] &
        df["close_recovery_ok"] &
        df["index_ok"]
    )

    # extract signals
    signals = df[df["take_trade"]].copy()
    signals = signals.reset_index(drop=True)
    signals.to_csv(out_signals_path, index=False)

    print(f"\nSaved {len(signals)} signals → {out_signals_path}")
    return signals


if __name__ == "__main__":
    csv_path = r"C:\PERSONAL_DATA\Startups\Stocks\Jim_Simons_Trading_Strategy\Heuristic\dev\data\all_stocks_full_suite_v2.csv"
    signals = apply_strategy_c_mean_reversion_relaxed(csv_path)
