import os

# Define the folder structure
folder_structure = {
    "trading_bot/": [
        "data/",
        "models/",
        "regimes/",
        "strategies/",
        "stock_selection/",
        "backtesting/",
        "execution/",
        "visualization/",
    ],
    "trading_bot/regimes/": [
        "bull_market.py",
        "bear_market.py",
        "mean_reversion.py",
        "volatility.py",
        "liquidity.py",
        "sentiment.py",
        "transition.py",
    ],
    "trading_bot/visualization/": [
        "regime_plot.py",
        "stock_performance.py",
        "backtest_results.py",
    ],
    "trading_bot/": [
        "main.py",
        "config.py",
        "utils.py",
    ],
}

# Create the folder structure
for folder, files in folder_structure.items():
    os.makedirs(folder, exist_ok=True)  # Create directories if they don't exist
    for file in files:
        with open(os.path.join(folder, file), "w") as f:
            f.write("# Placeholder file\n")  # Create empty files with a comment

print("✅ Trading bot folder structure created successfully!")
