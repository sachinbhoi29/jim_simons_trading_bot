# trading_bot.py
import config
from data.data_handler import DataHandler
from models.market_regime_detector import MarketRegimeDetector
from stock_selection.stock_selector import StockSelector
from strategies.strategy_executor import StrategyExecutor
from backtesting.backtester import Backtester
from execution.executor import Executor
from visualization.visualizer import Visualizer

class TradingBot:
    def __init__(self):
        self.config = config.load_config()
        self.data_handler = DataHandler()
        self.regime_detector = MarketRegimeDetector()
        self.stock_selector = StockSelector()
        self.strategy_executor = StrategyExecutor()
        self.backtester = Backtester()
        self.executor = Executor()
        self.visualizer = Visualizer()

    def run(self):
        """Main function to run the trading bot."""
        self.update_data()
        market_regime = self.detect_market_regime()
        selected_stocks = self.select_stocks(market_regime)
        strategy = self.select_strategy(market_regime)
        self.execute_strategy(strategy, selected_stocks)

    def update_data(self):
        """Fetches market data and updates databases."""
        self.data_handler.fetch_data()

    def detect_market_regime(self):
        """Detects the current market regime."""
        return self.regime_detector.predict_regime()

    def select_stocks(self, market_regime):
        """Filters stocks based on the detected market regime."""
        return self.stock_selector.select_stocks(market_regime)

    def select_strategy(self, market_regime):
        """Selects the best strategy for the current regime."""
        return self.strategy_executor.select_strategy(market_regime)

    def execute_strategy(self, strategy, selected_stocks):
        """Executes the selected trading strategy."""
        self.strategy_executor.execute(strategy, selected_stocks)

    def backtest_strategies(self):
        """Backtests all strategies."""
        self.backtester.run_backtest()

    def visualize_results(self):
        """Generates charts for analysis."""
        self.visualizer.plot_regime()
        self.visualizer.plot_stock_performance()
        self.visualizer.plot_backtest_results()

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()

