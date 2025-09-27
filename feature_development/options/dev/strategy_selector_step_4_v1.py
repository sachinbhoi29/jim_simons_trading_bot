import json
from pathlib import Path

# ===============================
# 1️⃣ Load market snapshot JSON
# ===============================
json_path = Path("feature_development/options/dev/market_parameters.json")
with open(json_path) as f:
    market_data = json.load(f)

# Extract key info
trend = market_data['market_snapshot'].get('trend', 'Neutral')               # Bullish / Bearish / Neutral / Delta Neutral
volatility = market_data['market_snapshot'].get('volatility_regime', 'Moderate')  # Low / Moderate / High

# ===============================
# 2️⃣ Define strategy library
# ===============================
STRATEGY_LIBRARY = {
    "Bullish": {
        "Low": [
            {"name": "Short Put", "description": "Sell out-of-the-money put"},
            {"name": "Covered Call", "description": "Buy stock + sell call"},
            {"name": "Bull Call Spread", "description": "Buy call + sell call at higher strike"},
            {"name": "Bull Put Spread", "description": "Sell put + buy lower strike put"},
            {"name": "Call Ratio Back Spread", "description": "Sell 1 call, buy 2 higher strike calls"}
        ],
        "Moderate": [
            {"name": "Long Call", "description": "Directional call"},
            {"name": "Call Debit Spread", "description": "Reduce cost vs long call"},
            {"name": "Bull Call Spread", "description": "Buy call + sell call at higher strike"},
            {"name": "Bull Put Spread", "description": "Sell put + buy lower strike put"},
            {"name": "Call Ratio Back Spread", "description": "Sell 1 call, buy 2 higher strike calls"}
        ],
        "High": [
            {"name": "Long Call", "description": "High volatility bullish"},
            {"name": "Ratio Call Spread", "description": "Reduce cost, risky if big move up"},
            {"name": "Bull Call Spread", "description": "Buy call + sell call at higher strike"},
            {"name": "Bull Put Spread", "description": "Sell put + buy lower strike put"},
            {"name": "Call Ratio Back Spread", "description": "Sell 1 call, buy 2 higher strike calls"}
        ]
    },
    "Bearish": {
        "Low": [
            {"name": "Short Call", "description": "Sell out-of-the-money call"},
            {"name": "Protective Put", "description": "Hedge long positions"},
            {"name": "Bear Call Spread", "description": "Sell call + buy higher strike call"},
            {"name": "Bear Put Spread", "description": "Buy put + sell lower strike put"},
            {"name": "Put Ratio Back Spread", "description": "Sell 1 put, buy 2 lower strike puts"}
        ],
        "Moderate": [
            {"name": "Long Put", "description": "Directional bearish"},
            {"name": "Put Debit Spread", "description": "Limit cost of bearish bet"},
            {"name": "Bear Call Spread", "description": "Sell call + buy higher strike call"},
            {"name": "Bear Put Spread", "description": "Buy put + sell lower strike put"},
            {"name": "Put Ratio Back Spread", "description": "Sell 1 put, buy 2 lower strike puts"}
        ],
        "High": [
            {"name": "Long Put", "description": "High volatility bearish"},
            {"name": "Ratio Put Spread", "description": "Reduce cost, risky if big move down"},
            {"name": "Bear Call Spread", "description": "Sell call + buy higher strike call"},
            {"name": "Bear Put Spread", "description": "Buy put + sell lower strike put"},
            {"name": "Put Ratio Back Spread", "description": "Sell 1 put, buy 2 lower strike puts"}
        ]
    },
    "Neutral": {
        "Low": [
            {"name": "Iron Condor", "description": "Defined risk, range-bound"},
            {"name": "Butterfly Spread", "description": "Low risk range-bound"},
            {"name": "Short Straddle", "description": "Sell call + put at same strike"},
            {"name": "Short Strangle", "description": "Sell OTM call + put"},
            {"name": "Long Iron Butterfly", "description": "Buy wings, sell ATM straddle"},
            {"name": "Short Iron Butterfly", "description": "Sell wings, buy ATM straddle"},
            {"name": "Long Iron Condor", "description": "Buy wings, sell middle strikes"},
            {"name": "Short Iron Condor", "description": "Sell wings, buy middle strikes"},
            {"name": "Call Calendar Spread", "description": "Sell near-term call, buy longer-term call"},
            {"name": "Put Calendar Spread", "description": "Sell near-term put, buy longer-term put"}
        ],
        "Moderate": [
            {"name": "Long Straddle", "description": "Profit from big move either side"},
            {"name": "Long Strangle", "description": "Less expensive, needs bigger move"},
            {"name": "Short Straddle", "description": "Sell call + put at same strike"},
            {"name": "Short Strangle", "description": "Sell OTM call + put"},
            {"name": "Long Iron Butterfly", "description": "Buy wings, sell ATM straddle"},
            {"name": "Short Iron Butterfly", "description": "Sell wings, buy ATM straddle"},
            {"name": "Long Iron Condor", "description": "Buy wings, sell middle strikes"},
            {"name": "Short Iron Condor", "description": "Sell wings, buy middle strikes"},
            {"name": "Call Calendar Spread", "description": "Sell near-term call, buy longer-term call"},
            {"name": "Put Calendar Spread", "description": "Sell near-term put, buy longer-term put"}
        ],
        "High": [
            {"name": "Long Straddle", "description": "Expect big moves"},
            {"name": "Protective Collar", "description": "Limited downside protection"},
            {"name": "Long Strangle", "description": "Less expensive, needs bigger move"},
            {"name": "Short Straddle", "description": "Sell call + put at same strike"},
            {"name": "Short Strangle", "description": "Sell OTM call + put"},
            {"name": "Long Iron Butterfly", "description": "Buy wings, sell ATM straddle"},
            {"name": "Short Iron Butterfly", "description": "Sell wings, buy ATM straddle"},
            {"name": "Long Iron Condor", "description": "Buy wings, sell middle strikes"},
            {"name": "Short Iron Condor", "description": "Sell wings, buy middle strikes"},
            {"name": "Call Calendar Spread", "description": "Sell near-term call, buy longer-term call"},
            {"name": "Put Calendar Spread", "description": "Sell near-term put, buy longer-term put"}
        ]
    },
    "Delta Neutral": {
        "Any": [
            {"name": "Short Straddle", "description": "Sell call + put at same strike"},
            {"name": "Long Straddle", "description": "Buy call + put at same strike"},
            {"name": "Short Strangle", "description": "Sell OTM call + put"},
            {"name": "Long Strangle", "description": "Buy OTM call + put"},
            {"name": "Long Iron Butterfly", "description": "Buy wings, sell ATM straddle"},
            {"name": "Short Iron Butterfly", "description": "Sell wings, buy ATM straddle"},
            {"name": "Long Iron Condor", "description": "Buy wings, sell middle strikes"},
            {"name": "Short Iron Condor", "description": "Sell wings, buy middle strikes"},
            {"name": "Call Calendar Spread", "description": "Sell near-term call, buy longer-term call"},
            {"name": "Put Calendar Spread", "description": "Sell near-term put, buy longer-term put"}
        ]
    }
}

# ===============================
# 3️⃣ Select strategies based on snapshot
# ===============================
if trend == "Delta Neutral":
    selected_strategies = STRATEGY_LIBRARY["Delta Neutral"]["Any"]
else:
    selected_strategies = STRATEGY_LIBRARY.get(trend, {}).get(volatility, [])

# ===============================
# 4️⃣ Build JSON output
# ===============================
output = {
    "market_snapshot": market_data['market_snapshot'],
    "selected_strategies": selected_strategies
}

# Save output
output_path = Path("feature_development/options/dev/strategy_selection.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=4)

print(f"✅ Strategy selection saved to {output_path}")
