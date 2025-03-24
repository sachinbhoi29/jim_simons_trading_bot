from pipeline.trading_bot_pipeline import pilotPipeline
import pandas as pd


if __name__ == "__main__":
    pp = pilotPipeline()
    # pp.get_data()
    pp.regime_detector()
    pp.markvo_chain()
    pp.indicators_calculation()
    pp.price_action()
    stock_nifty_combined = pp.merge_stock_with_nifty_regime()
    stock_nifty_combined.to_csv("data/sbin_nifty_combined.csv")


