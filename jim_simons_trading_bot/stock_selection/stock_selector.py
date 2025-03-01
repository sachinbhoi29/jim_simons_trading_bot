# stock_selection/stock_selector.py
import pandas as pd

class StockSelector:
    def __init__(self):
        self.stock_list = []

    def select_stocks(self, market_regime):
        """Filters stocks based on the detected market regime."""
        data = pd.read_csv("data/market_data.csv")
        if market_regime == "bull":
            return self.filter_by_momentum(data)
        elif market_regime == "bear":
            return self.filter_by_strength(data)
        else:
            return self.filter_by_liquidity(data)

    def filter_by_momentum(self, data):
        """Filters stocks with strong momentum."""
        return data[data["RSI"] > 50]

    def filter_by_strength(self, data):
        """Filters fundamentally strong stocks for defensive plays."""
        return data[data["PE Ratio"] < 20]

    def filter_by_liquidity(self, data):
        """Filters stocks with high liquidity."""
        return data[data["Volume"] > data["Volume"].mean()]
