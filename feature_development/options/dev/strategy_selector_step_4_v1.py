import json
from pathlib import Path

# ===============================
# 1️⃣ Load market snapshot JSON
# ===============================
json_path = Path("feature_development/options/dev/market_parameters.json")
with open(json_path) as f:
    market_data = json.load(f)

# Extract key info
trend = market_data['market_snapshot'].get('trend', 'Neutral')               # Bullish / Bearish / Neutral
volatility = market_data['market_snapshot'].get('volatility_regime', 'Moderate')  # Low / Moderate / High

# ===============================
# 2️⃣ Define strategy library
# ===============================
# Configurable: add more strategies under each market condition
STRATEGY_LIBRARY = {
    "Bullish": {
        "Low": [
            {"name": "Short Put", "description": "Sell out-of-the-money put"},
            {"name": "Covered Call", "description": "Buy stock + sell call"},
            {"name": "Bull Call Spread", "description": "Buy call + sell call at higher strike"}
        ],
        "Moderate": [
            {"name": "Long Call", "description": "Directional call"},
            {"name": "Call Debit Spread", "description": "Reduce cost vs long call"},
        ],
        "High": [
            {"name": "Long Call", "description": "High volatility bullish"},
            {"name": "Ratio Call Spread", "description": "Reduce cost, risky if big move up"},
        ]
    },
    "Bearish": {
        "Low": [
            {"name": "Short Call", "description": "Sell out-of-the-money call"},
            {"name": "Protective Put", "description": "Hedge long positions"},
        ],
        "Moderate": [
            {"name": "Long Put", "description": "Directional bearish"},
            {"name": "Put Debit Spread", "description": "Limit cost of bearish bet"},
        ],
        "High": [
            {"name": "Long Put", "description": "High volatility bearish"},
            {"name": "Ratio Put Spread", "description": "Reduce cost, risky if big move down"},
        ]
    },
    "Neutral": {
        "Low": [
            {"name": "Iron Condor", "description": "Defined risk, range-bound"},
            {"name": "Butterfly Spread", "description": "Low risk range-bound"},
        ],
        "Moderate": [
            {"name": "Long Straddle", "description": "Profit from big move either side"},
            {"name": "Long Strangle", "description": "Less expensive, needs bigger move"},
        ],
        "High": [
            {"name": "Long Straddle", "description": "Expect big moves"},
            {"name": "Protective Collar", "description": "Limited downside protection"},
        ]
    }
}

# ===============================
# 3️⃣ Select strategies based on snapshot
# ===============================
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
