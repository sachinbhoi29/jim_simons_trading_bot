import os
import numpy as np
import pandas as pd

def apply_strategy_c_mean_reversion_plus(
        csv_path,
        out_signals_path=r"C:\\PERSONAL_DATA\\Startups\\Stocks\\Jim_Simons_Trading_Strategy\\Heuristic\\dev\\data\\signals_only.csv",
        require_index_bullish=True):

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
    # STRATEGY C+  (Improved Mean Reversion)
    # ============================================================

    # --- Trend Filter (relaxed but safer)
    df["htf_trend_ok"] = df["EMA_20"] >= df["EMA_50"] * 0.985

    # --- Oversold conditions
    df["oversold_rsi"] = df["RSI_14"] < 45
    df["oversold_stoch"] = df["%K"] < 40

    # --- BB Oversold Zone (New)
    df["bb_oversold_zone"] = df["Close"] <= df["BB_lower_20"] * 1.02

    # --- Volume metrics
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["supply_exhaustion"] = df["Volume"] < df["vol_ma20"] * 0.9

    # --- Volatility flush
    df["atr_spike"] = df["ATR_14"] > df["ATR_14"].rolling(14).mean() * 1.05

    # --- Demand wick (New)
    df["demand_wick"] = (df["Close"] - df["Low"]) > (df["High"] - df["Close"])

    # --- Confirmation: need 2/3 triggers
    df["rsi_turn_up"] = df["RSI_14"] > df["RSI_14"].shift(1)
    df["stoch_turn_up"] = df["%K"] > df["%D"]
    df["close_up"] = df["Close"] > df["Close"].shift(1)

    df["confirmation_ok"] = (
        (df["rsi_turn_up"].astype(int) +
         df["stoch_turn_up"].astype(int) +
         df["close_up"].astype(int)) >= 2
    )

    # --- Index regime filter
    df["index_ok"] = True
    if require_index_bullish:
        df["index_ok"] = df["NIFTY_Trend"].str.contains("Bullish", case=False, na=False)

    # ============================================================
    # FINAL TRADE SIGNAL
    # ============================================================
    df["take_trade"] = (
        df["htf_trend_ok"] &
        df["oversold_rsi"] &
        df["oversold_stoch"] &
        df["bb_oversold_zone"] &
        df["supply_exhaustion"] &
        df["atr_spike"] &
        df["demand_wick"] &
        df["confirmation_ok"] &
        df["index_ok"]
    )

    signals = df[df["take_trade"]].copy()
    signals = signals.reset_index(drop=True)

    signals.to_csv(out_signals_path, index=False)
    print(f"\nSaved {len(signals)} signals → {out_signals_path}")

    return signals


if __name__ == "__main__":
    csv_path = r"C:\\PERSONAL_DATA\\Startups\\Stocks\\Jim_Simons_Trading_Strategy\\Heuristic\\dev\\data\\all_stocks_full_suite_v2.csv"
    signals = apply_strategy_c_mean_reversion_plus(csv_path)
