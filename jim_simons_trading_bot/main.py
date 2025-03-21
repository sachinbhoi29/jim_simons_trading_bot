from pipeline.trading_bot_pipeline import pilotPipeline


if __name__ == "__main__":
    pp = pilotPipeline()
    df = pp.regime_detector()
    pp.markvo_chain(df)

