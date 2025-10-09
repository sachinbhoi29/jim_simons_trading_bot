from pipeline.fib_strategy import ChartPipeline


if __name__ == '__main__':
    # Create a singleton instance for convenience
    pipeline = ChartPipeline()
    # Plot NIFTY and BANKNIFTY
    pipeline.plot("^NSEI")
    pipeline.plot(["^NSEBANK"])

    # # Filter tickers by Fib percent and save only if in 50-65%
    # tickers = ["RELIANCE.NS", "TCS.NS"]
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "LT.NS", "ITC.NS", "HINDUNILVR.NS", "HDFC.NS", "BAJFINANCE.NS", "MARUTI.NS", "ASIANPAINT.NS", "ULTRACEMCO.NS", "HCLTECH.NS", "WIPRO.NS", "TITAN.NS", "SUNPHARMA.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "BAJAJ-AUTO.NS", "HINDALCO.NS", "TATAMOTORS.NS", "M&M.NS", "DIVISLAB.NS", "NESTLEIND.NS", "JSWSTEEL.NS", "ADANIGREEN.NS", "HDFCLIFE.NS", "BPCL.NS", "EICHERMOT.NS", "SHREECEM.NS", "DRREDDY.NS", "CIPLA.NS", "TECHM.NS", "UPL.NS", "TATACONSUM.NS", "COALINDIA.NS", "GRASIM.NS", "INDUSINDBK.NS", "MARICO.NS", "TATAPOWER.NS", "HDFCAMC.NS", "INDIGO.NS", "MAXHEALTH.NS", "RECLTD.NS", "BANKBARODA.NS", "MUTHOOTFIN.NS", "M&MFIN.NS", "BHEL.NS", "BEL.NS", "HINDPETRO.NS", "IOC.NS", "NTPC.NS", "TATACHEM.NS", "HINDZINC.NS", "SAIL.NS", "NMDC.NS", "JSWENERGY.NS", "GAIL.NS", "TATAMETALI.NS", "JINDALSTEL.NS", "HINDCOPPER.NS", "INDIANB.NS", "PNB.NS", "RELIANCE.NS"]
    # fib_level_filter = [40, 60]
    # pipeline.strategy_1(tickers, fib_level_filter=fib_level_filter,start="2024-10-01",end="2025-05-01")
    fib_level_filter = [50, 61]
    pipeline.strategy_2(tickers, fib_level_filter=fib_level_filter,start="2024-02-01",end="2024-07-03")
