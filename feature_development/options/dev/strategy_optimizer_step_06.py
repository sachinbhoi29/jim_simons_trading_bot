import pandas as pd
from pathlib import Path
from strategies import IronCondorStrategy,optimize_strategy


if __name__ == "__main__":
    csv_path = Path("feature_development/options/dev/BANKNIFTY_options_30Sep2025_with_greeks.csv")
    df = pd.read_csv(csv_path)

    strategy = IronCondorStrategy()
    best_ic = optimize_strategy(df, strategy, top_n=3)
    for i, ic in enumerate(best_ic, 1):
        print(f"\n=== Iron Condor {i} ===")
        print("Short Put:", ic["legs"]["short_put"]["strike"])
        print("Long Put:", ic["legs"]["long_put"]["strike"])
        print("Short Call:", ic["legs"]["short_call"]["strike"])
        print("Long Call:", ic["legs"]["long_call"]["strike"])
        print("Net Score:", ic["net_score"])
        print("Net Delta:", ic["net_delta"])
        print("Net Theta:", ic["net_theta"])
        print("Net Vega:", ic["net_vega"])
