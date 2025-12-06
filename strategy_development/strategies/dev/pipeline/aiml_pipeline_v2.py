import os
import pandas as pd
import numpy as np
from get_data.yfinance_multistock_data import download_and_split, download_and_split_index
from overlays.indicators_v1 import (
    MovingAverageOverlay, RSIOverlay, VolumeOverlay, MACDOverlay,
    BollingerBandsOverlay, StochasticOscillatorOverlay, ATROverlay,
    EMAOverlay, FibonacciOverlay, VWAPOverlay
)
from overlays.zigzagsr_v1 import ZigzagSR
from overlays.regime_detection_v1 import EnhancedRegimeOverlay
from utilities.fib_utils import last_price_fib_info
from candelstick_base.candlestick_chart_v1 import CandlestickChart
from functools import reduce


# ----------------------------------------------------------
#   Utility: safe lookback percent features (no leakage)
# ----------------------------------------------------------
def compute_tp_sl_outcomes(df, horizon=5, tp_pct=0.02, sl_pct=0.01):
    """
    Day-by-day TP/SL detection over the next `horizon` trading rows.

    For each row i:
      - compute tp_level = Close_i * (1 + tp_pct)
      - compute sl_level = Close_i * (1 - sl_pct)
      - scan rows i+1 .. i+horizon (or until end)
          - if day j has High >= tp_level and tp_hit_day not set -> set tp_hit_day = (j - i)
          - if day j has Low  <= sl_level and sl_hit_day not set -> set sl_hit_day = (j - i)
      - future_tp_hit_day, future_sl_hit_day are 1-based day offsets (1..horizon) or NaN
      - future_bracket_outcome: "TP_HIT" | "SL_HIT" | "BOTH" | "NONE"
      - future_bracket_pl_pct: +tp_pct*100, -sl_pct*100, or 0
    """
    df = df.copy()

    # ensure datetime index and chronological order (oldest -> newest)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index(ascending=True)

    # Prepare arrays
    n = len(df)
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values

    tp_levels = closes * (1.0 + tp_pct)
    sl_levels = closes * (1.0 - sl_pct)

    tp_hit_day = np.full(n, np.nan)
    sl_hit_day = np.full(n, np.nan)
    outcome = np.full(n, None, dtype=object)
    pl_pct = np.full(n, np.nan)

    for i in range(n):
        tp_level = tp_levels[i]
        sl_level = sl_levels[i]

        # scan next rows day-by-day
        start = i + 1
        end = min(i + 1 + horizon, n)  # end is exclusive

        tp_day = None
        sl_day = None

        for j in range(start, end):
            offset = j - i  # 1-based offset
            h = highs[j]
            l = lows[j]

            # check TP
            if tp_day is None and (not pd.isna(h)) and h >= tp_level:
                tp_day = offset

            # check SL
            if sl_day is None and (not pd.isna(l)) and l <= sl_level:
                sl_day = offset

            # if both found, can break early (but keep values)
            if tp_day is not None and sl_day is not None:
                break

        tp_hit_day[i] = tp_day if tp_day is not None else np.nan
        sl_hit_day[i] = sl_day if sl_day is not None else np.nan

        # determine outcome
        if tp_day is None and sl_day is None:
            outcome[i] = "NONE"
            pl_pct[i] = 0.0
        elif tp_day is not None and sl_day is None:
            outcome[i] = "TP_HIT"
            pl_pct[i] = tp_pct * 100.0
        elif sl_day is not None and tp_day is None:
            outcome[i] = "SL_HIT"
            pl_pct[i] = -sl_pct * 100.0
        else:
            # both hit -> if same day mark BOTH; otherwise earliest wins
            if tp_day == sl_day:
                outcome[i] = "BOTH"
                pl_pct[i] = -sl_pct * 100.0  # follow your rule: use SL value when both same day
            else:
                if tp_day < sl_day:
                    outcome[i] = "TP_HIT"
                    pl_pct[i] = tp_pct * 100.0
                else:
                    outcome[i] = "SL_HIT"
                    pl_pct[i] = -sl_pct * 100.0

    # --- New PL_percent column ---
    pl_percent = np.full(n, np.nan)

    for i in range(n):
        if outcome[i] == "TP_HIT":
            pl_percent[i] = tp_pct * 100
        elif outcome[i] == "SL_HIT":
            pl_percent[i] = -sl_pct * 100
        else:
            pl_percent[i] = np.nan  # BOTH or NONE ignored


    # attach columns
    df['PL_percent'] = pl_percent
    df['tp_level'] = tp_levels
    df['sl_level'] = sl_levels
    df['future_tp_hit_day'] = tp_hit_day
    df['future_sl_hit_day'] = sl_hit_day
    df['future_bracket_outcome'] = outcome
    df['future_bracket_pl_pct'] = pl_pct

    return df



# ----------------------------------------------------------
#   Utility: compute forward/horizon stats and bracket labels
#   Supports both trading-day horizon (default) and calendar-day horizon.
# ----------------------------------------------------------
def compute_future_stats_and_bracket_labels(
    df, horizon=5, tp_pct=0.02, sl_pct=0.01, use_trading_days=True
):
    """
    Computes labels using future bars (this is correct for supervised learning).

    Parameters
    ----------
    df : DataFrame indexed by trading dates, must have 'High','Low','Close'
    horizon : int
        If use_trading_days=True -> number of trading rows ahead (typical)
        If use_trading_days=False -> number of calendar days ahead (will pick first trading row on/after date + horizon days)
    tp_pct, sl_pct : floats for bracket outcome detection
    use_trading_days : bool
        True -> horizon measured in trading rows (shift), False -> horizon as calendar days

    Returns
    -------
    df with columns added:
    ['future_return', 'future_max_high', 'future_min_low', 'future_max_pct', 'future_min_pct',
     'bracket_outcome', 'bracket_hit_day']
    """
    # ensure datetime index and chronological order
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index(ascending=True)

    n = len(df)
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    index = df.index

    future_return = np.full(n, np.nan)
    future_max_high = np.full(n, np.nan)
    future_min_low = np.full(n, np.nan)
    future_max_pct = np.full(n, np.nan)
    future_min_pct = np.full(n, np.nan)
    bracket_outcome = np.full(n, np.nan)  # 1 TP first, -1 SL first, 0 neither/ambiguous
    bracket_hit_day = np.full(n, np.nan)

    # helper to find calendar-target index (first trading index >= target_date)
    def find_target_idx_calendar(i, calendar_days):
        target_date = index[i] + pd.Timedelta(days=calendar_days)
        # get_indexer with method='bfill' returns -1 if beyond last index
        pos = index.get_indexer([target_date], method='bfill')[0]
        if pos == -1:
            return None
        return int(pos)

    for i in range(n):
        # determine the target index for the future_return
        if use_trading_days:
            target_idx = i + horizon
            if target_idx < n:
                future_return[i] = (closes[target_idx] / closes[i]) - 1.0
        else:
            t_idx = find_target_idx_calendar(i, horizon)
            if t_idx is not None and t_idx < n:
                future_return[i] = (closes[t_idx] / closes[i]) - 1.0
            # if calendar target missing (after end), leave NaN

        # compute future max/min over the next `horizon` trading rows (i+1 .. i+horizon)
        # bracket detection and future_max/min use actual trading rows after i.
        start_f = i + 1
        end_f = i + 1 + horizon
        if start_f < n:
            slice_end = min(end_f, n)
            window_highs = highs[start_f:slice_end]
            window_lows = lows[start_f:slice_end]

            if window_highs.size > 0:
                future_max_high[i] = np.max(window_highs)
            if window_lows.size > 0:
                future_min_low[i] = np.min(window_lows)

            if not np.isnan(future_max_high[i]):
                future_max_pct[i] = (future_max_high[i] / closes[i] - 1.0) * 100.0
            if not np.isnan(future_min_low[i]):
                future_min_pct[i] = (future_min_low[i] / closes[i] - 1.0) * 100.0

            # bracket logic: day-by-day hit order within the next `horizon` trading rows
            tp_price = closes[i] * (1.0 + tp_pct)
            sl_price = closes[i] * (1.0 - sl_pct)
            tp_hit_day = None
            sl_hit_day = None

            for j, (hval, lval) in enumerate(zip(window_highs, window_lows), start=1):
                if tp_hit_day is None and hval >= tp_price:
                    tp_hit_day = j
                if sl_hit_day is None and lval <= sl_price:
                    sl_hit_day = j
                if tp_hit_day is not None and sl_hit_day is not None:
                    break

            if tp_hit_day is not None and sl_hit_day is not None:
                if tp_hit_day < sl_hit_day:
                    bracket_outcome[i] = 1
                    bracket_hit_day[i] = tp_hit_day
                elif sl_hit_day < tp_hit_day:
                    bracket_outcome[i] = -1
                    bracket_hit_day[i] = sl_hit_day
                else:
                    # both on same future trading day — ambiguous: mark 0 (conservative)
                    bracket_outcome[i] = 0
                    bracket_hit_day[i] = tp_hit_day
            elif tp_hit_day is not None:
                bracket_outcome[i] = 1
                bracket_hit_day[i] = tp_hit_day
            elif sl_hit_day is not None:
                bracket_outcome[i] = -1
                bracket_hit_day[i] = sl_hit_day
            else:
                bracket_outcome[i] = 0
                bracket_hit_day[i] = np.nan

    # attach to df (use copy to avoid SettingWithCopy issues)
    df = df.copy()
    df['future_return'] = future_return
    df['future_max_high'] = future_max_high
    df['future_min_low'] = future_min_low
    df['future_max_pct'] = future_max_pct
    df['future_min_pct'] = future_min_pct
    df['bracket_outcome'] = bracket_outcome.astype('float')
    df['bracket_hit_day'] = bracket_hit_day

    return df


# ----------------------------------------------------------
#                    V2 Pipeline (corrected)
# ----------------------------------------------------------
class AIMLFeaturePipelineV2:
    def __init__(self, future_return=5,
                 data_dir="strategy_development/strategies/dev/data",
                 chart_dir="strategy_development/strategies/dev/chart",
                 use_trading_days=True,
                 live_prediction=False):
        """
        future_return : int -> horizon for labels (see use_trading_days)
        use_trading_days : bool -> if True horizon measured in trading rows (recommended).
                              if False horizon measured in calendar days (will map to next trading day).
        live_prediction : bool -> if True, future_return for the lookahead days will have 0,
        it will give only probablity for the next trading day, it is 0 because it is not yet known.
        """
        self.data_dir = data_dir
        self.chart_dir = chart_dir
        self.future_return = future_return
        self.use_trading_days = use_trading_days
        self.live_prediction = live_prediction   
        os.makedirs(self.data_dir, exist_ok=True)

    # ------------------------------------------------------
    #     STOCK-LEVEL FEATURES + BRACKET LABELS (ADDED)
    # ------------------------------------------------------
    def generate_stock_features(self, tickers, start=None, end=None, period=None,
                                add_index=False, show=False,
                                lookback=5, tp_pct=0.02, sl_pct=0.01):
        """
        Downloads tickers, computes overlays (features) and labels (future stats and bracket labels).
        All features are computed using data up to and including day i (no lookahead).
        Labels intentionally use future bars.
        """
        final_dfs = []

        if add_index:
            start_index = (pd.to_datetime(start) - pd.DateOffset(months=9)).strftime("%Y-%m-%d")
            df_index = self.generate_index_features(start=start_index, end=end, period=period)
        else:
            df_index = None

        dfs = download_and_split(tickers, start=start, end=end, period=period)

        for ticker, df in dfs.items():
            try:
                if df.empty or df.isna().all().all():
                    print(f"{ticker} returned empty or invalid data — skipped")
                    continue

                df = df.copy()
                df.columns.name = None
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                df['Ticker'] = ticker

                # basic immediate features
                df['Range_pct'] = (df['High'] - df['Low']) / df['Close']

                # compute overlays (these are features computed up to day i)
                chart = CandlestickChart(df, ticker=ticker, show_candles=True, show=show)

                # Trend & Regime
                chart.add_overlay(EMAOverlay(window=20, color="green", show=True))
                chart.add_overlay(EMAOverlay(window=50, color="red", show=True))
                chart.add_overlay(EMAOverlay(window=200, color="red", show=True))
                chart.add_overlay(BollingerBandsOverlay(show=True))
                chart.add_overlay(FibonacciOverlay(lookback=50, show=True))

                # Support/Resistance
                chart.add_overlay(ZigzagSR(
                    min_peak_distance=8, min_peak_prominence=10,
                    zone_merge_tolerance=0.007, max_zones=8,
                    color_zone="blue", alpha_zone=0.15,
                    show=True, show_fibo=False, show_trendline=True,
                    show_only_latest_fibo=False
                ))

                # Volume & Volatility
                chart.add_subplot(VolumeOverlay(), height_ratio=1)
                chart.add_subplot(VWAPOverlay(show=True), height_ratio=1)
                chart.add_subplot(ATROverlay(), height_ratio=1)

                # Momentum
                chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
                chart.add_subplot(MACDOverlay(show=True), height_ratio=1)
                chart.add_subplot(StochasticOscillatorOverlay(show=True), height_ratio=1)

                # extract features DataFrame (indicators computed up to day i)
                df_feat = chart.only_df()

                # merge index features if requested
                if add_index and df_index is not None and not df_index.empty:
                    df_index['Date_x'] = pd.to_datetime(df_index['Date_x'])
                    df_feat = df_feat.reset_index().rename(columns={'index': 'Date'})
                    df_feat = pd.merge(df_feat, df_index, left_on='Date', right_on='Date_x', how='left')
                    df_feat.set_index('Date', inplace=True)
                else:
                    # ensure proper index type
                    df_feat.index = pd.to_datetime(df_feat.index)

                # --- FIXED ORDER: compute future stats first (so future_max_high exists) ---
                df_feat = compute_future_stats_and_bracket_labels(
                    df_feat,
                    horizon=self.future_return,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    use_trading_days=self.use_trading_days
                )

                # then compute TP/SL outcome based on those future_* columns
                df_feat = compute_tp_sl_outcomes(df_feat, horizon=self.future_return, tp_pct=tp_pct, sl_pct=sl_pct)

                final_dfs.append(df_feat)

                # chart annotation & save (unchanged behavior)
                if show:
                    try:
                        fib_info = last_price_fib_info(df_feat)
                        fib_percent = round(fib_info.get("fib_percent", 0), 2)
                    except Exception:
                        fib_percent = 0.0

                    fibo_status = df_feat.get("Fibo_Status_Last_Close", pd.Series(["N/A"])).iloc[-1]
                    ema20 = df_feat["EMA_20"].iloc[-1]
                    ema50 = df_feat["EMA_50"].iloc[-1]
                    ema_trend = "Bullish" if ema20 > ema50 else "Bearish"
                    rsi = df_feat["RSI_14"].iloc[-1]
                    rsi_signal = "Oversold" if rsi < 40 else "Overbought" if rsi > 60 else "Neutral"
                    vwap = df_feat["VWAP"].iloc[-1]
                    close = df_feat["Close"].iloc[-1]
                    vol = df_feat["Volume"].iloc[-1]
                    avg_vol = df_feat["Volume"].rolling(20).mean().iloc[-1]
                    vol_signal = "High" if vol > avg_vol else "Low/Normal"

                    note_text = (
                        f"Fib%: {fib_percent}% ({fibo_status})\n"
                        f"EMA Trend: {ema_trend}\n"
                        f"RSI: {rsi:.1f} ({rsi_signal})\n"
                        f"VWAP: {'Above' if close > vwap else 'Below'}\n"
                        f"Volume: {vol_signal}"
                    )

                    chart.add_text(note_text, x=1, y=0.98, fontsize=11, color="black", bbox=True)
                    save_path = f"{self.chart_dir}/{ticker}_full_suite_v2.png"
                    chart.plot(save_path=save_path)
                    print(f"Saved V2 chart for {ticker} at {save_path}")

            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        if final_dfs:
            df_all = pd.concat(final_dfs, axis=0)

            # ------------------------------------------------------------
            #     REAL PREDICTION MODE: Fill all NaN future_return with 0
            # ------------------------------------------------------------
            if self.live_prediction:
                df_all['future_return'] = df_all['future_return'].fillna(0)

            out_path = f"{self.data_dir}/all_stocks_full_suite_v2.csv"
            df_all.to_csv(out_path, index=True)
            print(f"V2: Saved all tickers: {df_all.shape[0]} rows -> {out_path}")

    # ------------------------------------------------------
    #     Index Features (unchanged except structural safety)
    # ------------------------------------------------------
    def generate_index_features(self, start=None, end=None, period=None, show=False):
        tickers = ["^NSEI", "^NSEBANK"]
        dfs = download_and_split_index(tickers, start=start, end=end, period=period)

        index_features_list = []

        for ticker, df in dfs.items():
            if df.empty or df.isna().all().all():
                print(f"{ticker} returned empty or invalid data — skipped")
                continue

            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df = df.copy()

            chart = CandlestickChart(df, ticker=ticker, show_candles=False, show=show)
            chart.add_overlay(EMAOverlay(window=20, show=True))
            chart.add_overlay(EMAOverlay(window=50, show=True))
            chart.add_overlay(EMAOverlay(window=200, show=True))
            chart.add_subplot(RSIOverlay(period=14), height_ratio=1)
            chart.add_subplot(MACDOverlay(show=True), height_ratio=1)
            chart.add_subplot(ATROverlay(window=14), height_ratio=1)
            chart.add_overlay(BollingerBandsOverlay(show=True))
            chart.add_subplot(VolumeOverlay(), height_ratio=1)
            chart.add_subplot(VWAPOverlay(show=True), height_ratio=1)
            chart.add_subplot(StochasticOscillatorOverlay(show=True), height_ratio=1)
            chart.add_overlay(EnhancedRegimeOverlay(show=True))
            chart.plot()

            df["Date"] = df.index
            df_ind = chart.only_df()

            prefix = "NIFTY" if ticker == "^NSEI" else "BANKNIFTY"
            df_ind = df_ind.rename(columns={col: f"{prefix}_{col}" for col in df_ind.columns if col != "Date"})
            df_ind[f"{prefix}_Trend"] = np.where(
                df_ind[f"{prefix}_EMA_20"] > df_ind[f"{prefix}_EMA_50"], "Bullish", "Bearish"
            )

            index_features_list.append(df_ind)

        if index_features_list:
            df_index = reduce(lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True, how="outer"
            ), index_features_list)

            df_index.sort_index(inplace=True)
            df_index.rename(columns={'Date': 'Date_x'}, inplace=True)
            return df_index

        return None


if __name__ == '__main__':
    # Create a singleton instance for convenience
    # aiml_pipeline = AIMLFeaturePipeline(future_return=1)
    aiml_pipeline = AIMLFeaturePipelineV2(future_return=5)
    # Plot NIFTY and BANKNIFTY
    # fib_pipeline.plot("^NSEI",start="2023-02-01",end="2024-05-02")#period='1y')#,start='2024-01-01',end='2025-01-01')
    # pipeline.plot(["^NSEBANK"])

    # tickers = SMALL_CAP_TICKERS + LARGE_CAP_TICKERS + MID_CAP_TICKERS #["ABB.NS"] # at least 2 tickers
    tickers = BANK_STOCKS
    aiml_pipeline.generate_stock_features(tickers, start="2011-01-02",end="2014-12-21",add_index=True,show=False,lookback=5, tp_pct=0.02, sl_pct=0.01)
    # aiml_pipeline.generate_stock_features(tickers, start="2008-05-05", end="2016-09-08", add_index=True, show=False)
    # aiml_pipeline.generate_stock_features(tickers, start="2016-09-02", end="2025-11-15", add_index=True, show=False)
    # aiml_pipeline.generate_index_features(start="2022-02-01",end="2025-05-02")
