# check for future return value
# check for stocks selected and date range
from pipeline.fib_strategy import fibPipeline
from pipeline.generic_strategy import genericStrategyPipeline
# from pipeline.aiml_pipeline import AIMLFeaturePipeline
from pipeline.aiml_pipeline_v2 import AIMLFeaturePipelineV2
from config.config import SMALL_CAP_TICKERS, LARGE_CAP_TICKERS, MID_CAP_TICKERS    


if __name__ == '__main__':
    # Create a singleton instance for convenience
    # aiml_pipeline = AIMLFeaturePipeline(future_return=1)
    aiml_pipeline = AIMLFeaturePipelineV2(future_return=1,live_prediction=True)
    # Plot NIFTY and BANKNIFTY
    # fib_pipeline.plot("^NSEI",start="2023-02-01",end="2024-05-02")#period='1y')#,start='2024-01-01',end='2025-01-01')
    # pipeline.plot(["^NSEBANK"])

    tickers =  SMALL_CAP_TICKERS +LARGE_CAP_TICKERS + MID_CAP_TICKERS #["ABB.NS"] # at least 2 tickers
    aiml_pipeline.generate_stock_features(tickers, start="2025-05-02",end="2025-12-31",add_index=True,show=False,lookback=5, tp_pct=0.02, sl_pct=0.01)
    # aiml_pipeline.generate_stock_features(tickers, start="2008-05-05", end="2016-09-08", add_index=True, show=False)
    # aiml_pipeline.generate_stock_features(tickers, start="2016-09-02", end="2025-11-15", add_index=True, show=False)
    # aiml_pipeline.generate_index_features(start="2022-02-01",end="2025-05-02")


    