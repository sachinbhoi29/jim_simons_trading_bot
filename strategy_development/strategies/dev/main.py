from pipeline.fib_strategy import fibPipeline
from pipeline.generic_strategy import genericStrategyPipeline


if __name__ == '__main__':
    # Create a singleton instance for convenience
    fib_pipeline = fibPipeline()
    gen_pipeline = genericStrategyPipeline()
    # Plot NIFTY and BANKNIFTY
    # pipeline.plot("^NSEI",start='2024-01-01',end='2025-01-01')
    # pipeline.plot(["^NSEBANK"])

    # # # Filter tickers by Fib percent and save only if in 50-65%
    # # tickers = ["RELIANCE.NS", "TCS.NS"]
    tickers = [
        "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANIPOWER.NS", "ASIANPAINT.NS",
        "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS",
        "BANDHANBNK.NS", "BANKBARODA.NS", "BEL.NS", "BHARATFORG.NS", "BHARTIARTL.NS",
        "BIOCON.NS", "BHEL.NS", "BPCL.NS", "CIPLA.NS", "COALINDIA.NS", "CROMPTON.NS",
        "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GAIL.NS", "GRASIM.NS", "HAVELLS.NS",
        "HCLTECH.NS", "HDFC.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
        "HINDCOPPER.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "HINDZINC.NS", "ICICIBANK.NS",
        "ICICIPRULI.NS", "INDIANB.NS", "INDIGO.NS", "INDUSINDBK.NS", "INFY.NS", "IOC.NS",
        "ITC.NS", "JINDALSTEL.NS", "JSWENERGY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS",
        "LICI.NS", "LTI.NS", "LT.NS", "LUPIN.NS", "MARICO.NS", "MARUTI.NS", "MAXHEALTH.NS",
        "M&M.NS", "M&MFIN.NS", "MRF.NS", "MUTHOOTFIN.NS", "NESTLEIND.NS", "NMDC.NS",
        "NOCIL.NS", "NTPC.NS", "ONGC.NS", "PIDILITIND.NS", "PNB.NS", "POWERGRID.NS",
        "RECLTD.NS", "RELIANCE.NS", "SAIL.NS", "SBIN.NS", "SHREECEM.NS", "SUNPHARMA.NS",
        "SUNTV.NS", "TATACHEM.NS", "TATACONSUM.NS", "TATAMETALI.NS", "TATAMOTORS.NS",
        "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS",
        "UPL.NS", "ULTRACEMCO.NS", "ULTRATECH.NS", "WIPRO.NS",    "APOLLOHOSP.NS", 
        "BRITANNIA.NS","HEROMOTOCO.NS","SBILIFE.NS","POLYCAB.NS", "BATAINDIA.NS",
        "BALKRISIND.NS", "COROMANDEL.NS", "ACC.NS", "VOLTRONIC.NS", "TVSMOTOR.NS", 
        "JINDALSAW.NS", "CUMMINSIND.NS", "GODREJIND.NS""ABFRL.NS", "ABCAPITAL.NS", 
        "ALKEM.NS", "APLAPOLLO.NS", "ACC.NS", "BATAINDIA.NS", "BANKINDIA.NS", 
        "ASHOKLEY.NS", "ASTRAL.NS", "VOLTA.NS", "SUZLON.NS", "CUMMINSIND.NS", "COFORGE.NS", 
        "PERSISTENT.NS", "INDHOTEL.NS", "PBFINTECH.NS", "DLF.NS", "IPCA.NS", "TATACHEM.NS"]

    # fib_level_filter = [40, 60]
    # pipeline.strategy_1(tickers, fib_level_filter=fib_level_filter,start="2024-10-01",end="2025-05-01")
    fib_level_filter = [50, 61]
    fib_pipeline.strategy_2(tickers, fib_level_filter=fib_level_filter,start="2024-06-01",end="2024-11-12")
    # gen_pipeline.trend_fibo_conf_strategy(tickers,start="2024-06-01",end="2024-09-02")
    # gen_pipeline.multi_confluence_strategy(tickers,start="2024-06-01",end="2024-09-02")
    # gen_pipeline.strategy_volume_burst(tickers,start="2024-06-01",end="2024-09-02")