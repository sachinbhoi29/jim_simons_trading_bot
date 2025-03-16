from pipeline.trading_bot_pipeline import pilotPipeline


if __name__ == "__main__":
    pp = pilotPipeline()
    stock_data = pp.get_data()

