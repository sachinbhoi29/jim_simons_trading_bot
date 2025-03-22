from pipeline.trading_bot_pipeline import pilotPipeline
import pandas as pd


if __name__ == "__main__":
    pp = pilotPipeline()
    pp.regime_detector()
    pp.markvo_chain()
    pp.indicators_calculation()
    pp.price_action()

