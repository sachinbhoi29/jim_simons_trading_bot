import yfinance as yf
import matplotlib.pyplot as plt

# Download India VIX data for the last 6 months
india_vix = yf.download("^INDIAVIX", period="6mo", interval="1d")

# Drop any missing values
india_vix = india_vix.dropna()

# Plot India VIX with filter levels
plt.figure(figsize=(12, 6))
plt.plot(india_vix.index, india_vix['Close'], label='India VIX', color='purple')
plt.axhline(12, color='green', linestyle='--', label='Low Volatility Threshold (VIX=12)')
plt.axhline(16, color='blue', linestyle='--', label='Max VIX for Iron Condor (16)')
plt.axhline(18, color='orange', linestyle='--', label='No Straddles Above This (18)')
plt.axhline(25, color='red', linestyle='--', label='High Volatility Danger Zone (25)')
plt.title('India VIX - Last 6 Months')
plt.xlabel('Date')
plt.ylabel('VIX Level')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print most recent India VIX value
latest_vix = india_vix['Close'].iloc[-1]
latest_date = india_vix.index[-1]
print(f"Latest VIX (as of {latest_date.date()}): {latest_vix:.2f}")
