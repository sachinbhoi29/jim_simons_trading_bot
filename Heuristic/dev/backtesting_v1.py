import pandas as pd
import numpy as np

def simple_bracket_backtest(csv_path, direction="long", starting_capital=1_000_000):
    """
    Capital-based bracket-order backtest (fixed).
    For each entry DATE, allocates current equity equally across all signals that day,
    computes PnL for each trade using allocated capital (fractional shares allowed),
    and updates equity after all trades for that date complete.
    BOTH and NONE outcomes are excluded from P&L but included in diagnostics.
    """

    df = pd.read_csv(csv_path)

    # Required columns from your dataset
    required_cols = [
        "Date", "Adj Close", "Close", "High", "Low", "Open", "Volume", "Ticker", "Range_pct",
        "future_return", "future_max_high", "future_min_low", "future_max_pct", "future_min_pct",
        "bracket_outcome", "bracket_hit_day", "PL_percent",
        "tp_level", "sl_level",
        "future_tp_hit_day", "future_sl_hit_day", "future_bracket_outcome", "future_bracket_pl_pct"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Ensure Date is datetime and sort
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    # Keep a copy of original column names, then rename for internal usage
    df = df.copy()
    df = df.rename(columns={
        "Close": "entry_price",
        "future_bracket_outcome": "hit_result"
    })

    # Initialize new columns
    df["shares"] = np.nan
    df["trade_pnl_rupees"] = np.nan
    df["pnl_percent"] = np.nan
    df["cumulative_equity"] = np.nan
    df["allocated_capital"] = np.nan  # capital allocated to that trade

    equity = float(starting_capital)
    all_trade_pnls = []

    # Stats (diagnostics)
    tp_count = (df["hit_result"] == "TP_HIT").sum()
    sl_count = (df["hit_result"] == "SL_HIT").sum()
    both_hit_count = (df["hit_result"] == "BOTH").sum()
    none_hit_count = (df["hit_result"] == "NONE").sum()
    total_signals = len(df)

    # Group by entry date and process each day as a batch
    for date, group in df.groupby("Date", sort=True):
        # indices in original df
        idxs = group.index.tolist()

        # valid trades for PnL: only TP_HIT or SL_HIT
        valid_mask = group["hit_result"].isin(["TP_HIT", "SL_HIT"])
        valid_idxs = group[valid_mask].index.tolist()
        n_valid = len(valid_idxs)

        # If no valid trades that day: record equity for those rows and continue
        if n_valid == 0:
            df.loc[idxs, "cumulative_equity"] = equity
            continue

        # Allocate equity equally across valid trades on this date
        # (You can change allocation rule later)
        alloc_per_trade = equity / n_valid

        # Track daily pnl to update equity after processing all trades that day
        daily_pnl_sum = 0.0

        for idx in valid_idxs:
            row = df.loc[idx]
            hit = row["hit_result"]
            entry = row["entry_price"]
            tp = row["tp_level"]
            sl = row["sl_level"]

            # Sanity checks
            if pd.isna(entry) or entry <= 0:
                # skip invalid entry prices (record cumulative equity)
                df.loc[idx, "cumulative_equity"] = equity
                df.loc[idx, "allocated_capital"] = np.nan
                continue

            # allocated capital for this trade
            allocated = alloc_per_trade

            # compute shares to buy (fractional allowed)
            shares = allocated / entry

            # determine exit price
            if direction == "long":
                exit_price = tp if hit == "TP_HIT" else sl
                trade_pnl_rupees = shares * (exit_price - entry)
                pnl_percent = (exit_price - entry) / entry * 100.0
            else:  # short case (kept simple)
                # for short: profit when price falls
                exit_price = tp if hit == "TP_HIT" else sl
                trade_pnl_rupees = shares * (entry - exit_price)
                pnl_percent = (entry - exit_price) / entry * 100.0

            # Save results into df (allocated, shares, pnl)
            df.loc[idx, "allocated_capital"] = allocated
            df.loc[idx, "shares"] = shares
            df.loc[idx, "trade_pnl_rupees"] = trade_pnl_rupees
            df.loc[idx, "pnl_percent"] = pnl_percent

            daily_pnl_sum += trade_pnl_rupees
            all_trade_pnls.append(trade_pnl_rupees)

        # After all trades for the date are computed, update equity once
        equity += daily_pnl_sum

        # write same-day cumulative_equity for all rows (including invalid and BOTH/NONE)
        df.loc[idxs, "cumulative_equity"] = equity

    # Convert results to series
    rupee_results = pd.Series(all_trade_pnls) if len(all_trade_pnls) else pd.Series(dtype=float)
    percent_results = df["pnl_percent"].dropna()

    # Calculate expectancy
    expectancy_rupees = rupee_results.mean() if len(rupee_results) > 0 else 0
    expectancy_percent = percent_results.mean() if len(percent_results) > 0 else 0

    # Summary
    pnl_summary = {
        "Starting Capital (₹)": starting_capital,
        "Ending Capital (₹)": equity,
        "Net Profit (₹)": equity - starting_capital,
        "Return (%)": round((equity - starting_capital) / starting_capital * 100, 2),
        "Total Trades Counted": len(rupee_results),
        "Winning Trades": (rupee_results > 0).sum(),
        "Losing Trades": (rupee_results < 0).sum(),
        "Win Rate (%)": round((rupee_results > 0).mean() * 100, 2) if len(rupee_results) else 0,
        "Total PnL (₹)": rupee_results.sum(),
        "Average PnL per Trade (₹)": expectancy_rupees,
        "Max Win (₹)": rupee_results.max() if len(rupee_results) else 0,
        "Max Loss (₹)": rupee_results.min() if len(rupee_results) else 0,
        "Expectancy (₹)": expectancy_rupees,
        "Expectancy per Trade (%)": round(expectancy_percent, 6),
    }

    diagnostics = {
        "Total Signals": total_signals,
        "TP Hits": tp_count,
        "SL Hits": sl_count,
        "Both Hit": both_hit_count,
        "None Hit": none_hit_count,
        "TP Hit (%)": round(tp_count / total_signals * 100, 2),
        "SL Hit (%)": round(sl_count / total_signals * 100, 2),
        "Both Hit (%)": round(both_hit_count / total_signals * 100, 2),
        "None Hit (%)": round(none_hit_count / total_signals * 100, 2),
    }

    return pnl_summary, diagnostics, rupee_results, df


# ───────────────────────────────────────────────—
# MAIN EXECUTION
# ───────────────────────────────────────────────—
if __name__ == "__main__":
    print("=== Simple Bracket Backtester (fixed allocation per entry date) ===")

    csv_path = r"C:\PERSONAL_DATA\Startups\Stocks\Jim_Simons_Trading_Strategy\Heuristic\dev\data\signals_only.csv"
    print(f"Using CSV file:\n{csv_path}\n")

    pnl_summary, diagnostics, trades, df_clean = simple_bracket_backtest(csv_path)

    print("=== PnL Summary ===")
    for k, v in pnl_summary.items():
        print(f"{k}: {v}")

    print("\n=== Trade Outcome Stats ===")
    for k, v in diagnostics.items():
        print(f"{k}: {v}")

    output_path = r"C:\PERSONAL_DATA\Startups\Stocks\Jim_Simons_Trading_Strategy\Heuristic\dev\data\backtesting.csv"
    df_clean.to_csv(output_path, index=False)

    print(f"\nBacktest output saved to: {output_path}")
    print("\nDone.")
