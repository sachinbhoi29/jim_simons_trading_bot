import pandas as pd

INPUT_CSV = r"C:\PERSONAL_DATA\Startups\Stocks\Jim_Simons_Trading_Strategy\AI_ML\ML\dev_v4\data\Ensemble_highconf_trades_prediction_only_stats.csv"
OUTPUT_CSV = r"C:\PERSONAL_DATA\Startups\Stocks\Jim_Simons_Trading_Strategy\AI_ML\ML\dev_v4\data\model_returns_each_day.csv"

df = pd.read_csv(INPUT_CSV)

# Convert future_return into percent
df["future_return"] = df["future_return"] * 100

# Create pivot table
pivot = pd.pivot_table(
    df,
    index='Date',
    values='future_return',
    aggfunc=['sum', 'count']
).reset_index()

# Flatten & rename columns
pivot.columns = ['Date', 'future_return_sum_percent', 'num_trades']

# Add average return per trade
pivot['avg_return_per_trade_percent'] = pivot['future_return_sum_percent'] / pivot['num_trades']

# Save only the pivot table
pivot.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print(pivot.head())
