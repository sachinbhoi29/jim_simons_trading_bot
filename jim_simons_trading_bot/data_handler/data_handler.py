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
from nselib import capital_market


# Load configuration
config = configparser.ConfigParser()
config.read("config.ini")


class DataSource(ABC):
    """Abstract Base Class for different data sources."""
    @abstractmethod
    def fetch_data(self, symbol, period, interval):
        pass


PROXY_LIST_URL = "https://www.proxy-list.download/api/v1/get?type=https"

def get_free_proxies():
    """Fetch free HTTPS proxies from an online source."""
    try:
        response = requests.get(PROXY_LIST_URL, timeout=10)
        if response.status_code == 200:
            proxies = response.text.split("\r\n")
            return [proxy for proxy in proxies if proxy.strip()]
        else:
            print("⚠️ Failed to fetch free proxies.")
            return []
    except Exception as e:
        print(f"❌ Error fetching proxies: {e}")
        return []    


class YahooFinanceDataSource(DataSource):
    """Concrete Implementation using Yahoo Finance with Optional Proxy Support."""
    def __init__(self):
        self.use_proxy = config["DATASOURCE"].getboolean("use_proxy", fallback=True)  # Default is True

    def fetch_data(self, symbol, period, interval):
        print(f"📡 Fetching market data from Yahoo Finance for {symbol}...")

        session = requests.Session()

        # Use proxies if enabled in config
        if self.use_proxy:
            proxies = get_free_proxies()
            random.shuffle(proxies)  # Shuffle proxies to avoid detection
            
            for proxy in proxies:
                try:
                    print(f"🔄 Trying Proxy: {proxy}")
                    session.proxies = {
                        "http": f"http://{proxy}",
                        "https": f"https://{proxy}"
                    }

                    df = yf.download(f"{symbol}.NS", period=period, interval=interval, session=session)

                    if not df.empty:
                        print(f"✅ Data fetched successfully for {symbol} using proxy!")
                        df["Symbol"] = symbol  # Add column to indicate stock
                        return df
                    else:
                        print(f"⚠️ No data returned for {symbol}, trying next proxy...")
                
                except requests.exceptions.RequestException:
                    print(f"🚫 Proxy {proxy} failed. Trying another one...")

            print(f"❌ All proxies failed for {symbol}. Consider using a VPN or waiting before retrying.")
            return None
        
        else:
            # Fetch without proxy
            print("🚀 Fetching data without a proxy...")
            try:
                df = yf.download(f"{symbol}.NS", period=period, interval=interval)
                if not df.empty:
                    print(f"✅ Data fetched successfully for {symbol} without a proxy!")
                    df["Symbol"] = symbol
                    return df
                else:
                    print(f"⚠️ No data returned for {symbol}.")
            except Exception as e:
                print(f"❌ Error fetching data for {symbol}: {e}")
            
            return None


class NSEIndiaDataSource(DataSource):
    """Concrete Implementation using NSE India API."""
    NSE_URL = "https://www.nseindia.com/api/quote-equity"

    def fetch_data(self, symbol, period, interval):
        print(f"Fetching market data from NSE India for {symbol}...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        params = {"symbol": symbol}
        response = requests.get(self.NSE_URL, headers=headers, params=params)

        if response.status_code == 200:
            json_data = response.json()
            if "priceInfo" in json_data:
                df = pd.DataFrame([json_data["priceInfo"]])
                df["Symbol"] = symbol
                print(f"✅ Data fetched successfully for {symbol}!")
                return df
            else:
                print(f"⚠️ No price data found for {symbol}.")
        else:
            print(f"⚠️ Error fetching data for {symbol}. HTTP {response.status_code}")
        
        return None


class NSELibDataSource(DataSource):
    """Concrete Implementation using NSElib for NSE India market data."""
    
    def convert_period(self, period):
        """Convert Yahoo Finance-style periods (1mo, 3mo) to NSElib-compatible periods (1M, 3M)."""
        conversion_map = {
            "1mo": "1M","3mo": "3M","6mo": "6M","1y": "1Y","2y": "2Y","5y": "5Y"}
        return conversion_map.get(period.lower(), period)  # Default to the same value if not found

    def fetch_data(self, symbol, period, interval, data_type="price_volume_and_deliverable_position_data"):
        """
        Fetches market data using NSElib.

        Parameters:
        - symbol (str): Stock symbol (e.g., "SBIN")
        - period (str): Time period ("1M", "3M", etc.)
        - interval (str): Time interval (not required for NSElib)
        - data_type (str): Type of data to fetch (default: price_volume_and_deliverable_position_data)
        """
        print(f"📡 Fetching {data_type} data from NSElib for {symbol}...")

        try:
            # Convert period format to NSElib-compatible format
            nse_period = self.convert_period(period)

            if data_type == "price_volume_and_deliverable_position_data":
                df = capital_market.price_volume_and_deliverable_position_data(symbol=symbol, period=nse_period)
            elif data_type == "bulk_deal_data":
                df = capital_market.bulk_deal_data()
            elif data_type == "block_deals_data":
                df = capital_market.block_deals_data()
            elif data_type == "bhav_copy_equities":
                df = capital_market.bhav_copy_equities()
            elif data_type == "index_data":
                df = capital_market.index_data()
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
            
            if df is not None and not df.empty:
                print(f"✅ Successfully fetched {data_type} data for {symbol}!")
                df["Symbol"] = symbol  # Add stock symbol column
                return df
            else:
                print(f"⚠️ No data returned for {symbol}.")
        
        except Exception as e:
            print(f"❌ Error fetching {data_type} data from NSElib for {symbol}: {e}")
        
        return None




CACHE_LOG_FILE = "cache_log.csv"

class DataHandler:
    def __init__(self):
        self.symbols = config["DATA"]["symbols"].split(", ")
        self.period = config["DATA"]["period"]
        self.interval = config["DATA"]["interval"]
        self.provider = config["DATASOURCE"]["provider"].strip().lower()

        # Choose data source strategy
        if self.provider == "yfinance":
            self.data_source = YahooFinanceDataSource()
        elif self.provider == "nseindia":
            self.data_source = NSEIndiaDataSource()
        elif self.provider == "nselib":
            self.data_source = NSELibDataSource()
        else:
            raise ValueError(f"Unknown data provider: {self.provider}")

        self.cache_log = self.load_cache_log()

    def load_cache_log(self):
        """Load the cache log file, or create a new one if it doesn't exist."""
        if os.path.exists(CACHE_LOG_FILE):
            return pd.read_csv(CACHE_LOG_FILE, index_col=0)
        else:
            return pd.DataFrame(columns=["symbol", "period", "interval", "filename", "last_updated"])

    def save_cache_log(self):
        """Save the cache log to a CSV file."""
        self.cache_log.to_csv(CACHE_LOG_FILE)

    def is_data_up_to_date(self, symbol):
        """Check if cached data is from today or last market close."""
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

    def is_last_market_day(self, date_str):
        """Check if the provided date is the last market open day."""
        given_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        today = datetime.date.today()
        weekday = today.weekday()

        # If today is Monday, last market day was Friday
        if weekday == 0 and given_date == today - datetime.timedelta(days=3):
            return True
        # If today is any other day, last market day was yesterday
        elif weekday > 0 and given_date == today - datetime.timedelta(days=1):
            return True
        
        return False

    def fetch_and_store_data(self):
        """Fetches market data for each stock and saves it separately, using cache when possible."""
        today = datetime.datetime.today().strftime("%Y-%m-%d")

        for symbol in self.symbols:
            print(f"📊 Processing stock: {symbol}")

            if self.is_data_up_to_date(symbol):
                print(f"✅ Using cached data for {symbol} (up-to-date).")
                continue  # Skip downloading if data is fresh

            df = self.data_source.fetch_data(symbol, self.period, self.interval)

            if df is not None:
                filename = f"data/{symbol}_{self.interval}_{self.period}.csv"
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

    def load_data(self):
        """Loads stored market data from CSV files, fetching new data if necessary."""
        stock_data = {}
        today = datetime.datetime.today().strftime("%Y-%m-%d")

        for symbol in self.symbols:
            filename = f"data/{symbol}_{self.interval}_{self.period}.csv"
            # If file exists and is up to date, then just read it, else if not fresh or data not available then download it and do it for each stock
            if os.path.exists(filename) and self.is_data_up_to_date(symbol):
                print(f"📂 Loading cached data for {symbol} from {filename}")
                stock_data[symbol] = pd.read_csv(filename)
            else:
                print(f"⚠️ No up-to-date cached data found for {symbol}. Fetching new data...")
                df = self.data_source.fetch_data(symbol, self.period, self.interval)
                if df is not None:
                    df.to_csv(filename, index=True)
                    stock_data[symbol] = df

                    # Update cache log
                    self.cache_log = self.cache_log[
                        ~((self.cache_log["symbol"] == symbol) & (self.cache_log["period"] == self.period) & (self.cache_log["interval"] == self.interval))
                    ]  # Remove old entry if it exists
                    new_entry = pd.DataFrame([[symbol, self.period, self.interval, filename, today]],
                                             columns=["symbol", "period", "interval", "filename", "last_updated"])
                    self.cache_log = pd.concat([self.cache_log, new_entry], ignore_index=True)
                    self.save_cache_log()

        return stock_data
    
# Example usage:
if __name__ == "__main__":
    handler = DataHandler()
    handler.fetch_and_store_data()
    stock_data = handler.load_data()

    for symbol, df in stock_data.items():
        print(f"\n📌 {symbol} Data Sample:")
        print(df.head())  # Display first few rows
