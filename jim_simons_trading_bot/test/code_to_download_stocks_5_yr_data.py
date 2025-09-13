import yfinance as yf
import pandas as pd

# Define symbol and fetch 5 years of daily data
symbol = "HDFCBANK.NS"  # .NS for NSE
sbin_data = yf.download(symbol, period="5y", interval="1d")

# Optional: Add symbol column
sbin_data["Symbol"] = "HDFCBANK"

# Display the first few rows
print(sbin_data.head())

# Save to CSV if needed
sbin_data.to_csv("HDFC_5yr_daily.csv")
