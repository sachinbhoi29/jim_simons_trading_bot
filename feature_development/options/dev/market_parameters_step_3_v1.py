import yfinance as yf
import datetime
import pandas as pd
import json
import numpy as np

# Define symbols
symbol = "^NSEI"          # Nifty 50
vix_symbol = "^INDIAVIX"  # India VIX

# Set time window
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=90)

# Download data
nifty = yf.download(symbol, start=start_date, end=end_date, progress=False, multi_level_index=False)
vix = yf.download(vix_symbol, start=start_date, end=end_date, progress=False, multi_level_index=False)

# Calculate indicators
nifty['EMA20'] = nifty['Close'].ewm(span=20, adjust=False).mean()
nifty['EMA50'] = nifty['Close'].ewm(span=50, adjust=False).mean()

delta = nifty['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
nifty['RSI'] = 100 - (100 / (1 + rs))

nifty.dropna(inplace=True)

latest_price = nifty['Close'].iloc[-1]
ema_20 = nifty['EMA20'].iloc[-1]
ema_50 = nifty['EMA50'].iloc[-1]
rsi = nifty['RSI'].iloc[-1]
vix_latest = vix['Close'].dropna().iloc[-1]


def next_trading_day(date):
    next_day = date + datetime.timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += datetime.timedelta(days=1)
    return next_day


today = datetime.datetime.today()
next_trade_day = next_trading_day(today)
next_trade_weekday = next_trade_day.weekday()

weekday_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][next_trade_weekday]

if next_trade_weekday in [0, 4]:
    day_note = "Monday and Friday - Higher volatility and gaps possible. Consider tighter stops."
else:
    day_note = "Midweek trading - better liquidity and less gaps."

if (latest_price > ema_20 > ema_50) and (rsi > 60):
    trend = 'Bullish'
elif (latest_price < ema_20 < ema_50) and (rsi < 40):
    trend = 'Bearish'
else:
    trend = 'Neutral'

def convert_np(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # converts np.bool_, np.int64, np.float64 to native
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")



output_data = {
    "market_snapshot": {
        "date": next_trade_day.strftime('%Y-%m-%d'),
        "weekday": weekday_name,
        "spot_price": latest_price,
        "vix": vix_latest,
        "ema20": ema_20,
        "ema50": ema_50,
        "rsi": rsi,
        "trend": trend,
        "volatility_regime": ("Low" if vix_latest < 12 else "Moderate" if vix_latest < 18 else "High"),
        "day_note": day_note,
        "indicators": {
            "EMA20_gt_EMA50": latest_price > ema_20 > ema_50,
            "EMA20_lt_EMA50": latest_price < ema_20 < ema_50,
            "RSI_overbought": rsi > 70,
            "RSI_oversold": rsi < 30
        },
        "volume_info": {
            "latest_volume": nifty['Volume'].iloc[-1],
            "avg_volume_20": nifty['Volume'].rolling(20).mean().iloc[-1],
            "avg_volume_50": nifty['Volume'].rolling(50).mean().iloc[-1]
        },
        "atr_14": ((nifty['High'] - nifty['Low']).rolling(14).mean()).iloc[-1],
        "roc_5": nifty['Close'].pct_change(5).iloc[-1]
    },
    "strategy_params": {
        "delta_target": {"min": 0.10, "max": 0.30},
        "dte_target": {"min": 7, "max": 45},
        "min_premium": 20,
        "risk_free_rate": 0.065,
        "dividend_yield": 0.0,
        "volatility": vix_latest
    },
    "technical_indicators": {
    "EMA_diff": ema_20 - ema_50,
    "EMA20_over_EMA50_pct": (ema_20 - ema_50)/ema_50,
    "MACD": (nifty['Close'].ewm(span=12).mean() - nifty['Close'].ewm(span=26).mean()).iloc[-1],
    "MACD_signal": (nifty['Close'].ewm(span=12).mean() - nifty['Close'].ewm(span=26).mean()).ewm(span=9).mean().iloc[-1],
    "Bollinger_upper": (nifty['Close'].rolling(20).mean() + 2*nifty['Close'].rolling(20).std()).iloc[-1],
    "Bollinger_lower": (nifty['Close'].rolling(20).mean() - 2*nifty['Close'].rolling(20).std()).iloc[-1],
    "Bollinger_width_pct": ((2*nifty['Close'].rolling(20).std()).iloc[-1] / ema_20)
},
"volatility_info": {
    "atr_pct": ((nifty['High'] - nifty['Low']).rolling(14).mean() / latest_price).iloc[-1],
    "roc_5_pct": nifty['Close'].pct_change(5).iloc[-1],
    "vix_change_1d": vix['Close'].pct_change().iloc[-1]
},
"liquidity_info": {
    "volume_ratio_20_50": nifty['Volume'].rolling(20).mean().iloc[-1] / nifty['Volume'].rolling(50).mean().iloc[-1]
},

}


with open("feature_development/options/dev/market_parameters.json", "w") as f:
    json.dump(output_data, f, default=convert_np, indent=4)