from data_handler.data_handler import DataHandler
from regimes.market_regime_detector import MarketRegimeDetector
from regimes.markov_chain_and_ml import MarketRegimeForecaster

class pilotPipeline:
    def __init__(self):
        pass
    
    def get_data(self):
        handler = DataHandler()
        stock_data = handler.load_data()        
        return stock_data
    
    def regime_detector(self):
        detector = MarketRegimeDetector(file_path="data/NIFTY50_1d_5y.csv")
        regime_df = detector.run_regime_detector()
        return regime_df

    def markvo_chain(self,df):
        forecaster = MarketRegimeForecaster(df)
        forecaster.run_forecast_pipeline()