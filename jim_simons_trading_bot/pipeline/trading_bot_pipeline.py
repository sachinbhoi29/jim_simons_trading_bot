from data_handler.data_handler import DataHandler
from regimes.market_regime_detector import MarketRegimeDetector
from regimes.markov_chain_and_ml import MarketRegimeForecaster
from features_calculations.indicators_base_class import IndicatorBase
from features_calculations.price_action_base_class import PriceActionBase
import pandas as pd
from utils.common_functions import merge_stock_with_nifty_regime,normalize_ohlcv_columns


stock_to_analyze = 'data/ICICIBANK_4h_1y.csv'
class pilotPipeline:
    def __init__(self):
        pass
    
    def get_data(self):  # Sadly my IP is blocked by yfinance, and I have no idea why
        print("Part 1: Regime Detection")
        print("Step 1: Getting data from Yahoo Finance...")
        handler = DataHandler()
        stock_data = handler.load_data()        
        return stock_data
    
    def regime_detector(self):
        print("Step 2: Detecting market regimes...")
        detector = MarketRegimeDetector(file_path="data/NIFTY50_1d_5y.csv")
        self.regimes_df = detector.run_regime_detector()

    def markvo_chain(self):
        print("Step 3: Forecasting future market regimes...")
        forecaster = MarketRegimeForecaster(regimes_df = self.regimes_df,combined_actual_and_forecast_file_name="combined_actual_and_forecast.csv")
        self.nifty_combined_actual_and_forecast_df = forecaster.run_forecast_pipeline()
    
    def indicators_calculation(self,stock_df= None):
        print("Part 2: Stock Feature Calculations")
        print("Step 4: Calculating technical indicators...")
        stock_df = pd.read_csv(stock_to_analyze)
        stock_df = normalize_ohlcv_columns(stock_df)
        if stock_df.empty:
            print("Error: stock_df is empty. Please provide a valid DataFrame.")
            return
        indicators = IndicatorBase(stock_df)
        self.df_indicators = indicators.compute_indicators()

    def price_action(self,df_indicators=None):
        print("Step 5: Analyzing price actions...")
        pa = PriceActionBase(self.df_indicators)
        self.df_stock_indicators_and_price_action = pa.compute_candlestick_patterns()
        self.df_stock_indicators_and_price_action = pa.compute_support_resistance(window=20)
        self.df_stock_indicators_and_price_action = pa.compute_context_flags(tolerance=0.015)
        return self.df_stock_indicators_and_price_action

    def merge_stock_with_nifty_regime(self):
        print("Part 3: Merging Data")
        print("Step 6: Merging stock data with NIFTY50 regime data...")
        self.df_stock_regime = merge_stock_with_nifty_regime(self.df_stock_indicators_and_price_action, self.nifty_combined_actual_and_forecast_df)
        return self.df_stock_regime
