import yfinance as yf
import datetime
import pandas as pd
import json

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


strategies = []

if vix_latest < 12:
    # Low volatility
    if trend == 'Bullish':
        strategies = [
            {
                'name': 'Bull Put Spread (Credit Spread)',
                'prob_success': '65-70%',
                'premium_target': '₹40-60',
                'notes': 'Defined risk, good for stable uptrends with low volatility.'
            },
            {
                'name': 'Covered Call',
                'prob_success': '60-65%',
                'premium_target': '₹30-50',
                'notes': 'Income generation in bullish scenario with limited upside.'
            },
            {
                'name': 'Cash-Secured Put',
                'prob_success': '60-65%',
                'premium_target': '₹40-60',
                'notes': 'Generate income, ready to buy stock if assigned.'
            },
        ]
    elif trend == 'Bearish':
        strategies = [
            {
                'name': 'Bear Call Spread (Credit Spread)',
                'prob_success': '65-70%',
                'premium_target': '₹40-60',
                'notes': 'Limited risk bearish strategy in low volatility.'
            },
            {
                'name': 'Protective Put',
                'prob_success': '60-65%',
                'premium_target': 'Variable',
                'notes': 'Hedge long position during bearish outlook.'
            },
            {
                'name': 'Short Call',
                'prob_success': '60-65%',
                'premium_target': '₹40-60',
                'notes': 'Income strategy with risk if market rises sharply.'
            },
        ]
    else:
        strategies = [
            {
                'name': 'Iron Condor',
                'prob_success': '70-75%',
                'premium_target': '₹60-80',
                'notes': 'Range-bound strategy with defined risk.'
            },
            {
                'name': 'Calendar Spread',
                'legs': f"Sell near-term options, Buy longer-term options at same strikes",
                'prob_success': '65-70%',
                'premium_target': '₹40-60',
                'notes': 'Profit from time decay in range market.'
            },
            {
                'name': 'Butterfly Spread',
                'legs': f"Buy 1 ITM, Sell 2 ATM, Buy 1 OTM options",
                'prob_success': '65-70%',
                'premium_target': '₹40-60',
                'notes': 'Low risk, limited profit range-bound strategy.'
            },
        ]
elif vix_latest < 18:
    # Moderate volatility
    if trend == 'Bullish':
        strategies = [
            {
                'name': 'Long Call',
                'prob_success': '50-55%',
                'premium_target': '₹40-60',
                'notes': 'Directional bullish play; higher reward but higher risk.'
            },
            {
                'name': 'Call Debit Spread',
                'prob_success': '55-60%',
                'premium_target': '₹40-60',
                'notes': 'Limits cost/risk vs pure long call.'
            },
            {
                'name': 'Bull Put Spread',
                'prob_success': '60-65%',
                'premium_target': '₹40-60',
                'notes': 'Credit spread with defined risk, bullish bias.'
            },
        ]
    elif trend == 'Bearish':
        strategies = [
            {
                'name': 'Long Put',
                'prob_success': '50-55%',
                'premium_target': '₹40-60',
                'notes': 'Directional bearish play; high reward and risk.'
            },
            {
                'name': 'Put Debit Spread',
                'prob_success': '55-60%',
                'premium_target': '₹40-60',
                'notes': 'Limits cost/risk compared to pure put.'
            },
            {
                'name': 'Bear Call Spread',
                'prob_success': '60-65%',
                'premium_target': '₹40-60',
                'notes': 'Credit spread with bearish bias and defined risk.'
            },
        ]
    else:
        strategies = [
            {
                'name': 'Long Straddle',
                'prob_success': '50%',
                'premium_target': '₹60-80',
                'notes': 'Profits from big moves either way; watch time decay.'
            },
            {
                'name': 'Long Strangle',
                'prob_success': '50%',
                'premium_target': '₹60-80',
                'notes': 'Less expensive than straddle; needs bigger move.'
            },
            {
                'name': 'Protective Collar',
                'prob_success': '60-65%',
                'premium_target': 'Variable',
                'notes': 'Limits downside with limited upside; good in uncertain market.'
            },
        ]
else:
    # High volatility (>18)
    if trend == 'Bullish':
        strategies = [
            {
                'name': 'Long Call',
                'prob_success': '50-55%',
                'premium_target': '₹40-60',
                'notes': 'Directional bullish with expensive options, watch volatility crush.'
            },
            {
                'name': 'Ratio Call Spread',
                'prob_success': '55-60%',
                'premium_target': '₹40-60',
                'notes': 'Reduce cost but risk if big move up.'
            },
            {
                'name': 'Bull Call Spread',
                'prob_success': '60%',
                'premium_target': '₹40-60',
                'notes': 'Limited risk/reward bullish spread.'
            },
        ]
    elif trend == 'Bearish':
        strategies = [
            {
                'name': 'Long Put',
                'prob_success': '50-55%',
                'premium_target': '₹40-60',
                'notes': 'Directional bearish with expensive options.'
            },
            {
                'name': 'Ratio Put Spread',
                'prob_success': '55-60%',
                'premium_target': '₹40-60',
                'notes': 'Reduce cost, risk if big move down.'
            },
            {
                'name': 'Bear Put Spread',
                'prob_success': '60%',
                'premium_target': '₹40-60',
                'notes': 'Limited risk/reward bearish spread.'
            },
        ]
    else:
        strategies = [
            {
                'name': 'Long Straddle',
                'prob_success': '50%',
                'premium_target': '₹60-80',
                'notes': 'Expect big moves, avoid selling premium.'
            },
            {
                'name': 'Long Strangle',
                'prob_success': '50%',
                'premium_target': '₹60-80',
                'notes': 'Less expensive than straddle, benefits from volatility.'
            },
            {
                'name': 'Protective Collar',
                'prob_success': '60-65%',
                'premium_target': 'Variable',
                'notes': 'Limited downside protection in volatile markets.'
            },
        ]

# Print summary and strategies
print("=== MARKET SNAPSHOT ===")
print(f"Date: {next_trade_day.strftime('%Y-%m-%d')} ({weekday_name})")
print(f"Nifty Spot: {latest_price:.2f}")
print(f"VIX: {vix_latest:.2f}")
print(f"EMA-20: {ema_20:.2f}, EMA-50: {ema_50:.2f}")
print(f"RSI: {rsi:.2f}")
print(f"Market Trend: {trend}")
print(f"Volatility Regime: {'Low' if vix_latest < 12 else 'Moderate' if vix_latest < 18 else 'High'}")
print(f"\nMarket notes: {day_note}\n")

print("=== STRATEGY OPTIONS ===")
for i, strat in enumerate(strategies, 1):
    print(f"{i}. {strat['name']}")
    print(f"   Approx. Probability of Success: {strat['prob_success']}")
    print(f"   Target Premium: {strat['premium_target']}")
    print(f"   Notes: {strat['notes']}\n")


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
        "volatility_regime": (
            "Low" if vix_latest < 12 else
            "Moderate" if vix_latest < 18 else
            "High"
        ),
        "day_note": day_note
    },
    "strategies": strategies
}

# Convert dict to JSON string (pretty print)
json_output = json.dumps(output_data, indent=4)

with open("feature_development/options/dev/strategy_output.json", "w") as f:
    json.dump(output_data, f, indent=4)
