import pandas as pd
import yfinance as yf
import os
import configparser
import requests
import time
import random
from abc import ABC, abstractmethod
import datetime
# Import NSElib
# from nselib import capital_market, derivatives


# Load configuration
config = configparser.ConfigParser()
config.read("config/config.ini")
CACHE_LOG_FILE = "log/cache_log.csv"


class DataSource(ABC):
    """Abstract Base Class for different data sources."""
    
    @abstractmethod
    def fetch_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Fetches market data for a given stock symbol.

        Parameters:
        - symbol (str): Stock symbol (e.g., "RELIANCE").
        - period (str): Time range of data (e.g., "1mo", "3mo", "6mo", "1y").
        - interval (str): Time interval of data (e.g., "1d", "1h", "5m").

        Returns:
        - DataFrame: Contains market data with Open, High, Low, Close, Volume.
        """
        pass


class YahooFinanceDataSource:
    """Concrete Implementation using Yahoo Finance without Proxy Support."""
    
    def __init__(self):
        """Initialize Yahoo Finance data source."""
        print("📡 Yahoo Finance Data Source Initialized.")

    def fetch_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Fetches market data from Yahoo Finance.

        Parameters:
        - symbol (str): Stock symbol (e.g., "RELIANCE").
        - period (str): Time range of data (e.g., "1mo", "3mo", "1y").
        - interval (str): Time interval of data (e.g., "1d", "1h").

        Returns:
        - DataFrame: Contains stock data with OHLC and volume.
        """
        print(f"🚀 Fetching data for {symbol} from Yahoo Finance...")

        try:
            self.symbol_mapping = {
            "NIFTY50": "^NSEI",
            "SENSEX": "^BSESN",
            "BANKNIFTY": "^NSEBANK"
            }
            if not symbol in self.symbol_mapping:
                # Fetch data using yfinance
                df = yf.download(f"{symbol}.NS", period=period, interval=interval)
            else:
                index = yf.Ticker(f"{self.symbol_mapping[symbol]}")
                df = index.history(period=period)  # Get 1-year data

            if not df.empty:
                print(f"✅ Data fetched successfully for {symbol}!")
                df["Symbol"] = symbol  # Add symbol column
                return df
            else:
                print(f"⚠️ No data returned for {symbol}.")
                return None

        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None


class NSELibDataSource(DataSource):
    """Concrete Implementation using NSElib for NSE India market data."""
    
    def convert_period(self, period: str) -> str:
        """
        Converts Yahoo Finance-style periods to NSElib-compatible periods.

        Parameters:
        - period (str): Yahoo Finance-style period (e.g., "1mo", "3mo").

        Returns:
        - str: NSElib-compatible period (e.g., "1M", "3M").
        """
        conversion_map = {
            "1mo": "1M","3mo": "3M","6mo": "6M","1y": "1Y","2y": "2Y","5y": "5Y"}
        return conversion_map.get(period.lower(), period)  # Default to the same value if not found

    def fetch_data(self, symbol: str, period: str, interval: str, data_type="price_volume_and_deliverable_position_data") -> pd.DataFrame:
        """
        Fetches market data using NSElib.

        Parameters:
        - symbol (str): Stock or Index symbol (e.g., "SBIN", "NIFTY50", "BANKNIFTY").
        - period (str): Time range (e.g., "1M", "3M").
        - interval (str): Time interval (ignored for NSElib).
        - data_type (str): Type of data to fetch (default: price and volume).

        Returns:
        - DataFrame: Contains stock or index data.
        """
        print(f"📡 Fetching {data_type} data from NSElib for {symbol}...")

        try:
            # Convert period format to NSElib-compatible format
            nse_period = self.convert_period(period)

            # ✅ If the symbol is an index, fetch index data
            if symbol in ["NIFTY50", "SENSEX", "BANKNIFTY"]:
                index_mapping = {
                    "NIFTY50": "NIFTY 50",
                    "BANKNIFTY": "NIFTY BANK",
                    "SENSEX": "SENSEX"
                }
                df = capital_market.index_data(index=index_mapping[symbol])  # ✅ Correct function

            # ✅ Fetch stock data for individual stocks
            elif data_type == "price_volume_and_deliverable_position_data":
                df = capital_market.price_volume_and_deliverable_position_data(symbol=symbol, period=nse_period)

            elif data_type == "bulk_deal_data":
                df = capital_market.bulk_deal_data()
            elif data_type == "block_deals_data":
                df = capital_market.block_deals_data()
            elif data_type == "bhav_copy_equities":
                df = capital_market.bhav_copy_equities()
            elif data_type == "market_watch_all_indices":
                df = capital_market.market_watch_all_indices()
            elif data_type == "nse_live_option_chain":
                df = capital_market.nse_live_option_chain(symbol=symbol)
            elif data_type == "future_price_volume_data":
                df = capital_market.future_price_volume_data(symbol=symbol, instrument='FUTSTK', period=nse_period)
            elif data_type == "option_price_volume_data":
                df = capital_market.option_price_volume_data(symbol=symbol, instrument='OPTSTK', period=nse_period)
            else:
                print(f"⚠️ Data type {data_type} not recognized.")
                return None

            # ✅ Check if data is available
            if df is not None and not df.empty:
                print(f"✅ Successfully fetched {data_type} data for {symbol}!")
                df["Symbol"] = symbol  # Add stock symbol column
                return df
            else:
                print(f"⚠️ No data returned for {symbol}.")
        
        except Exception as e:
            print(f"❌ Error fetching {data_type} data from NSElib for {symbol}: {e}")
        
        return None 


class DataHandler:
    """Handles fetching, storing, and loading of stock market data from various sources."""
    def __init__(self):
        """Initialize the data handler with configuration settings and cache log."""
        self.symbols = config["DATA"]["symbols"].split(", ")
        self.period = config["DATA"]["period"]
        self.interval = config["DATA"]["interval"]
        self.provider = config["DATASOURCE"]["provider"].strip().lower()

        # Choose data source strategy
        if self.provider == "yfinance":
            self.data_source = YahooFinanceDataSource()
        # elif self.provider == "nseindia":
        #     self.data_source = NSEIndiaDataSource()
        elif self.provider == "nselib":
            self.data_source = NSELibDataSource()
        else:
            raise ValueError(f"Unknown data provider: {self.provider}")

        self.cache_log = self.load_cache_log()

    def load_cache_log(self) -> pd.DataFrame:
        """
        Load the cache log file, or create a new one if it doesn't exist.

        Returns:
        - pd.DataFrame: Cache log containing previously fetched stock data.
        """
        """Load the cache log file, or create a new one if it doesn't exist."""
        if os.path.exists(CACHE_LOG_FILE):
            return pd.read_csv(CACHE_LOG_FILE, index_col=0)
        else:
            return pd.DataFrame(columns=["symbol", "period", "interval", "filename", "last_updated"])

    def save_cache_log(self) -> None:
        """
        Save the cache log to a CSV file.
        """
        print(f"💾 Cache log saved to {CACHE_LOG_FILE}")
        self.cache_log.to_csv(CACHE_LOG_FILE)

    def is_data_up_to_date(self, symbol: str) -> bool:
        """
        Check if cached data is from today or last market close.

        Parameters:
        - symbol (str): Stock symbol (e.g., "RELIANCE").

        Returns:
        - bool: True if data is up to date, otherwise False.
        """
        today = datetime.datetime.today().strftime("%Y-%m-%d")

        # Find matching record in cache log
        cached_data = self.cache_log[
            (self.cache_log["symbol"] == symbol) &
            (self.cache_log["period"] == self.period) &
            (self.cache_log["interval"] == self.interval)
        ]

        if not cached_data.empty:
            last_updated = cached_data.iloc[0]["last_updated"]

            # If data is from today or last market close, consider it valid
            return last_updated == today or self.is_last_market_day(last_updated)
        
        return False  # No valid cached data found

    def is_last_market_day(self, date_str: str) -> bool:
        """
        Check if the provided date is the last market open day.

        Parameters:
        - date_str (str): Date string in "YYYY-MM-DD" format.

        Returns:
        - bool: True if it is the last market day, otherwise False.
        """
        try:
            given_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            given_date = datetime.datetime.strptime(date_str, "%d-%m-%Y").date()
        today = datetime.date.today()
        weekday = today.weekday()

        # If today is Monday, last market day was Friday
        if weekday == 0 and given_date == today - datetime.timedelta(days=3):
            return True
        # If today is any other day, last market day was yesterday
        elif weekday > 0 and given_date == today - datetime.timedelta(days=1):
            return True
        
        return False

    def fetch_and_store_data(self) -> None:
        """
        Fetches market data for each stock and saves it separately, using cache when possible.
        """
        today = datetime.datetime.today().strftime("%Y-%m-%d")

        for symbol in self.symbols:
            print(f"📊 Processing stock: {symbol}")

            if self.is_data_up_to_date(symbol):
                print(f"✅ Using cached data for {symbol} (up-to-date).")
                continue  # Skip downloading if data is fresh

            df = self.data_source.fetch_data(symbol, self.period, self.interval)

            if df is not None:
                filename = f"data/{symbol}_{self.interval}_{self.period}.csv"
                print(f"💾 Saving data for {symbol} to {filename}")
                df.to_csv(filename, index=True)
                print(f"💾 Data for {symbol} saved as {filename}")

                # Update cache log
                self.cache_log = self.cache_log[
                    ~((self.cache_log["symbol"] == symbol) & (self.cache_log["period"] == self.period) & (self.cache_log["interval"] == self.interval))
                ]  # Remove old entry if it exists
                new_entry = pd.DataFrame([[symbol, self.period, self.interval, filename, today]],
                                         columns=["symbol", "period", "interval", "filename", "last_updated"])
                self.cache_log = pd.concat([self.cache_log, new_entry], ignore_index=True)
                self.save_cache_log()
            else:
                print(f"⚠️ Skipping {symbol} due to data fetch failure.")

    def load_data(self) -> dict:
        """
        Loads stored market data from CSV files, fetching new data if necessary.

        Returns:
        - dict: A dictionary containing DataFrames with stock data and metadata.
        """
        stock_data = {}  # Dictionary to store stock data and metadata
        today = datetime.datetime.today().strftime("%Y-%m-%d")

        for symbol in self.symbols:
            filename = f"data/{symbol}_{self.interval}_{self.period}.csv"

            # Check if cached data exists for this stock
            cached_data = self.cache_log[
                (self.cache_log["symbol"] == symbol) &
                (self.cache_log["period"] == self.period) &
                (self.cache_log["interval"] == self.interval)
            ]

            if os.path.exists(filename) and not cached_data.empty:
                last_updated = cached_data.iloc[0]["last_updated"]
                last_updated = datetime.datetime.strptime(last_updated, "%d-%m-%Y").strftime("%Y-%m-%d")
                
                # If data is from today or last market close, load from cache
                if last_updated == today or self.is_last_market_day(last_updated):
                    print(f"📂 Loading cached data for {symbol} from {filename} (Last Updated: {last_updated})")
                    df = pd.read_csv(filename)
                    stock_data[symbol] = {
                        "data": df,
                        "metadata": {
                            "last_updated": last_updated,
                            "source": filename}}
                    continue  # Skip fetching new data if cache is valid
            # If no valid cache, fetch new data
            print(f"⚠️ No up-to-date cached data found for {symbol}. Fetching new data...")
            df = self.data_source.fetch_data(symbol, self.period, self.interval)

            if df is not None:
                print(f"💾 Data for {symbol} saved as {filename}")
                df.to_csv(filename, index=True)
                stock_data[symbol] = {
                    "data": df,
                    "metadata": {
                        "last_updated": today,
                        "source": filename
                    }
                }

                # Update cache log
                self.cache_log = self.cache_log[
                    ~((self.cache_log["symbol"] == symbol) & (self.cache_log["period"] == self.period) & (self.cache_log["interval"] == self.interval))
                ]  # Remove old entry if it exists
                new_entry = pd.DataFrame([[symbol, self.period, self.interval, filename, today]],
                                        columns=["symbol", "period", "interval", "filename", "last_updated"])
                self.cache_log = pd.concat([self.cache_log, new_entry], ignore_index=True)
                self.save_cache_log()

        return stock_data  # Returns a dictionary with DataFrames and metadata

    
# Example usage:
if __name__ == "__main__":
    handler = DataHandler()
    stock_data = handler.load_data()

    print('stock_data',stock_data)
    # Access Reliance's market data
    reliance_data = stock_data.get("RELIANCE")

    if reliance_data:
        print("\n📊 Reliance Market Data:")
        print(reliance_data["data"].head())  # Show first few rows of the DataFrame
        print("\n📋 Metadata:")
        print(reliance_data["metadata"])  # Show last updated date and source
    else:
        print("⚠️ Reliance data not available.")


