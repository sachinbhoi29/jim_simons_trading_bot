from data_handler.data_handler import DataHandler
from regimes.market_regime_detector import MarketRegimeDetector
from regimes.markov_chain_and_ml import MarketRegimeForecaster
from features_calculations.indicators_base_class import IndicatorBase
from features_calculations.price_action_base_class import PriceActionBase
import pandas as pd

class pilotPipeline:
    def __init__(self):
        pass
    
    def get_data(self):  # Sadly my IP is blocked by yfinance, and I have no idea why
        print("Step 1: Getting data from Yahoo Finance...")
        handler = DataHandler()
        stock_data = handler.load_data()        
        return stock_data
    
    def regime_detector(self):
        print("Step 2: Detecting market regimes...")
        detector = MarketRegimeDetector(file_path="data/NIFTY50_1d_5y.csv")
        self.regime_df = detector.run_regime_detector()

    def markvo_chain(self):
        print("Step 3: Forecasting future market regimes...")
        forecaster = MarketRegimeForecaster(self.regime_df)
        forecaster.run_forecast_pipeline()
    
    def indicators_calculation(self,stock_df= None):
        print("Step 4: Calculating technical indicators...")
        if stock_df.isempty:
            print("Error: stock_df is empty. Please provide a valid DataFrame.")
            return
        indicators = IndicatorBase(stock_df)
        self.df_indicators = indicators.compute_indicators()
        # return self.df_indicators  # in case you only needs indicators

    def price_action(self,df_indicators):
        print("Step 5: Analyzing price actions...")
        pa = PriceActionBase(self.df_indicators)
        self.df_indicators_and_price_action = pa.compute_candlestick_patterns()
        self.df_indicators_and_price_action = pa.compute_support_resistance(window=20)
        self.df_indicators_and_price_action = pa.compute_context_flags(tolerance=0.015)
        return self.df_indicators_and_price_action