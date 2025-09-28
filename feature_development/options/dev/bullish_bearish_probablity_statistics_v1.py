import pandas as pd
import numpy as np
import yfinance as yf

# -------------------------
# Utility: simple sigmoid
# -------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------------------------
# 1) Pivot finder (simple)
# -------------------------
def find_pivots(df, left=5, right=5, column="Close"):
    """
    Mark pivot highs and lows.
    A pivot high is where value is max in [i-left, i+right], pivot low is min in that window.
    Returns two boolean Series: is_pivot_high, is_pivot_low
    """
    series = df[column]
    n = len(series)
    is_high = pd.Series(False, index=series.index)
    is_low = pd.Series(False, index=series.index)

    # Avoid edges
    for i in range(left, n - right):
        window = series.iloc[i-left:i+right+1]
        center = series.iloc[i]
        if center == window.max():
            is_high.iloc[i] = True
        if center == window.min():
            is_low.iloc[i] = True

    return is_high, is_low

# -------------------------------------
# 2) Build S/R levels from pivots
# -------------------------------------
def get_support_resistance(df, lookback=250, left=5, right=5, n_levels=5):
    """
    Compute recent pivot highs/lows and return support/resistance levels (most recent n_levels).
    Returns dict with lists: 'supports', 'resistances' (ordered newest -> older).
    """
    is_high, is_low = find_pivots(df, left=left, right=right, column="Close")
    pivots = df.loc[is_high | is_low, ["High", "Low", "Close"]].copy()
    pivots["type"] = np.where(is_high.loc[pivots.index], "res", "sup")

    # restrict to lookback
    pivots = pivots.loc[df.index[-1] - pd.Timedelta(days=0):]  # placeholder (we'll slice by tail)
    pivots = pivots.tail(lookback)

    resistances = pivots.loc[pivots["type"] == "res", "High"].dropna().astype(float)
    supports = pivots.loc[pivots["type"] == "sup", "Low"].dropna().astype(float)

    # take most recent n_levels
    res_list = list(resistances.iloc[::-1].unique()[:n_levels])
    sup_list = list(supports.iloc[::-1].unique()[:n_levels])

    return {"supports": sup_list, "resistances": res_list}

# -------------------------------------
# 3) Indicator computations (VWAP, RSI, MACD, ATR)
# -------------------------------------
def compute_vwap(df):
    if "Volume" not in df.columns:
        raise ValueError("VWAP needs 'Volume' column")
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["Cum_TPV"] = (tp * df["Volume"]).cumsum()
    df["Cum_Vol"] = df["Volume"].cumsum()
    df["VWAP"] = df["Cum_TPV"] / df["Cum_Vol"]
    return df

def compute_rsi(df, period=14, price_col="Close", out_col=None):
    if out_col is None:
        out_col = f"RSI_{period}"
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df[out_col] = rsi
    return df

def compute_macd(df, fast=12, slow=26, signal=9):
    df["EMA_fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df["EMA_slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df

def compute_atr(df, window=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"ATR_{window}"] = tr.rolling(window=window, min_periods=1).mean()
    return df

# -------------------------------------
# 4) Build normalized features and probability model
# -------------------------------------
def compute_indicator_features(df):
    # Ensure necessary indicators exist. If not, compute them.
    if "VWAP" not in df.columns:
        df = compute_vwap(df)
    if "RSI_14" not in df.columns:
        df = compute_rsi(df, period=14)
    if "MACD" not in df.columns:
        df = compute_macd(df)
    if "ATR_14" not in df.columns:
        df = compute_atr(df, window=14)

    # feature values for last row
    last = df.iloc[-1]
    price = last["Close"]

    # VWAP position: normalized distance (positive => price > vwap)
    vwappos = (price - last["VWAP"]) / last["VWAP"]

    # RSI normalized (center at 50)
    rsi_norm = (last["RSI_14"] - 50) / 50  # roughly -1..1 for 0..100

    # MACD signal
    macd_val = last["MACD"]
    macd_sig = last["MACD_signal"]
    macd_cross = np.sign(macd_val - macd_sig)  # +1 bullish, -1 bearish, 0 neutral
    macd_magnitude = (macd_val - macd_sig) / (abs(price) + 1e-9)

    # ATR normalized by price (~volatility)
    atr_norm = last["ATR_14"] / price

    # Volume spike: ratio of last volume to rolling median
    if "Volume" in df.columns:
        vol_med = df["Volume"].rolling(window=20, min_periods=1).median().iloc[-1]
        vol_spike = (last["Volume"] / (vol_med + 1e-9)) - 1.0  # 0 means equal to median
    else:
        vol_spike = 0.0

    features = {
        "price": price,
        "vwappos": vwappos,
        "rsi_norm": rsi_norm,
        "macd_sign": macd_cross,
        "macd_magnitude": macd_magnitude,
        "atr_norm": atr_norm,
        "vol_spike": vol_spike
    }
    return features, df

# -------------------------------------
# 5) Probability function combining S/R + features
# -------------------------------------
def next_move_probability(df, weights=None, bias=0.0, lookback_pivots=250, pivot_left=5, pivot_right=5, n_levels=5):
    """
    Returns (df_with_cols, summary_dict) where df_with_cols has added columns for nearest S/R & probabilities.
    Probability model is a weighted linear combination of explainable features followed by sigmoid.
    """
    # default weights (tunable)
    if weights is None:
        weights = {
            "vwappos": 2.0,        # price > vwap pushes probability up
            "rsi_norm": 1.2,       # higher RSI -> more chance to continue up (but careful)
            "macd_sign": 1.0,      # MACD bullish -> favors up
            "macd_magnitude": 1.0, # magnitude adds conviction
            "atr_norm": -3.0,      # more ATR reduces confidence (more uncertainty)
            "vol_spike": 0.8,      # higher volume can strengthen directional bias
            "dist_support": 2.5,   # closer to support increases chance to go up
            "dist_resistance": -2.5 # closer to resistance decreases chance to go up
        }

    # compute indicators if missing
    feats, df = compute_indicator_features(df)

    # identify supports/resistances
    sr = get_support_resistance(df, lookback=lookback_pivots, left=pivot_left, right=pivot_right, n_levels=n_levels)
    supports = sr["supports"]
    resistances = sr["resistances"]
    price = feats["price"]

    # nearest support distance (positive if price above support; if no support, set large value)
    if supports:
        # use closest support below price; if none below, take the closest anyway
        sup_arr = np.array(supports, dtype=float)
        # compute distance as fraction of price: (price - support)/price ; if support > price -> negative
        dist_sups = (price - sup_arr) / price
        # choose the support that is <= price if possible
        below_mask = sup_arr <= price
        if any(below_mask):
            dist_to_support = np.min(dist_sups[below_mask])
        else:
            dist_to_support = np.min(dist_sups)  # negative, means support is above price (rare)
    else:
        dist_to_support = 1.0  # far

    # nearest resistance distance: (resistance - price)/price
    if resistances:
        res_arr = np.array(resistances, dtype=float)
        dist_res = (res_arr - price) / price
        # choose nearest resistance above price if possible
        above_mask = res_arr >= price
        if any(above_mask):
            dist_to_resistance = np.min(dist_res[above_mask])
        else:
            dist_to_resistance = np.min(dist_res)
    else:
        dist_to_resistance = 1.0

    # clip distances to reasonable range
    dist_to_support = float(np.clip(dist_to_support, -0.5, 0.5))
    dist_to_resistance = float(np.clip(dist_to_resistance, -0.5, 0.5))

    # build linear score
    linear = 0.0
    linear += weights["vwappos"] * feats["vwappos"]
    linear += weights["rsi_norm"] * feats["rsi_norm"]
    linear += weights["macd_sign"] * feats["macd_sign"]
    linear += weights["macd_magnitude"] * feats["macd_magnitude"]
    linear += weights["atr_norm"] * feats["atr_norm"]
    linear += weights["vol_spike"] * feats["vol_spike"]
    linear += weights["dist_support"] * (1 - abs(dist_to_support)) * (1 if dist_to_support >= 0 else -1)
    linear += weights["dist_resistance"] * (1 - abs(dist_to_resistance)) * (1 if dist_to_resistance <= 0 else -1)
    linear += bias

    prob_up = float(sigmoid(linear))
    prob_down = 1.0 - prob_up

    # Add columns for inspection
    df.loc[df.index[-1], "Prob_Up"] = prob_up
    df.loc[df.index[-1], "Prob_Down"] = prob_down
    df.loc[df.index[-1], "Nearest_Support"] = supports[0] if supports else np.nan
    df.loc[df.index[-1], "Nearest_Resistance"] = resistances[0] if resistances else np.nan
    df.loc[df.index[-1], "Dist_To_Support"] = dist_to_support
    df.loc[df.index[-1], "Dist_To_Resistance"] = dist_to_resistance
    df.loc[df.index[-1], "Model_Linear_Score"] = linear

    # Explanation dict
    explanation = {
        "prob_up": prob_up,
        "prob_down": prob_down,
        "linear_score": linear,
        "features": feats,
        "nearest_supports": supports,
        "nearest_resistances": resistances,
        "dist_to_support": dist_to_support,
        "dist_to_resistance": dist_to_resistance,
        "weights": weights,
        "bias": bias
    }

    # decision suggestion
    if prob_up > 0.65:
        suggestion = "Bias: LONG (high prob up)"
    elif prob_down > 0.65:
        suggestion = "Bias: SHORT (high prob down)"
    else:
        suggestion = "Neutral / Wait"

    explanation["suggestion"] = suggestion
    return df, explanation
if __name__ == "__main__":
    ticker = "^NSEBANK"
    start = "2024-01-01"
    end = "2025-08-21"
    df = yf.download(ticker,start=start,end=end,interval="1d",auto_adjust=False,multi_level_index=False)
    # df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False, multi_level_index=False)

    # compute overlays/indicators you already have
    df = compute_vwap(df)
    df = compute_rsi(df, period=14)
    df = compute_macd(df)
    df = compute_atr(df, window=14)

    # get next-move probability (last row)
    df, explanation = next_move_probability(df)

    print("Probability Up:", explanation["prob_up"])
    print("Probability Down:", explanation["prob_down"])
    print("Nearest Support(s):", explanation["nearest_supports"])
    print("Nearest Resistance(s):", explanation["nearest_resistances"])
    print("Suggestion:", explanation["suggestion"])
    print("Model linear score:", explanation["linear_score"])
    print("Features used:", explanation["features"])

    # you can inspect df.tail() for columns: VWAP, RSI_14, MACD, ATR_14, Prob_Up, Prob_Down etc.
