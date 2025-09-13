import mplfinance as mpf

def plot_candlestick_with_indicators(df, title="Candlestick with Indicators", window=100, save_path=None):
    df_plot = df.tail(window).copy()
    ohlc = df_plot[["Open", "High", "Low", "Close", "Volume"]].copy()
    apds = []

    panel_counter = 0  # panel 0 = price
    panel_ratios = [2]  # price panel

    # Volume is automatically assigned panel 1 when volume=True
    panel_counter += 1
    panel_ratios.append(0.5)  # smaller height for volume

    # Track additional panel numbers
    rsi_panel = None
    macd_panel = None

    # Add moving averages
    if "50EMA" in df_plot.columns:
        apds.append(mpf.make_addplot(df_plot["50EMA"], panel=0, color="blue", width=1.0))
    if "200EMA" in df_plot.columns:
        apds.append(mpf.make_addplot(df_plot["200EMA"], panel=0, color="purple", width=1.0))

    # Bollinger Bands
    if "BB_Upper" in df_plot.columns and "BB_Lower" in df_plot.columns:
        apds.append(mpf.make_addplot(df_plot["BB_Upper"], panel=0, color="grey", linestyle="--"))
        apds.append(mpf.make_addplot(df_plot["BB_Lower"], panel=0, color="grey", linestyle="--"))

    # RSI (optional panel)
    if "RSI" in df_plot.columns:
        panel_counter += 1
        rsi_panel = panel_counter
        apds.append(mpf.make_addplot(df_plot["RSI"], panel=rsi_panel, color='orange', ylabel='RSI'))
        panel_ratios.append(1)

    # MACD (optional panel)
    if "MACD" in df_plot.columns and "Signal" in df_plot.columns:
        panel_counter += 1
        macd_panel = panel_counter
        apds.append(mpf.make_addplot(df_plot["MACD"], panel=macd_panel, color='green', ylabel='MACD'))
        apds.append(mpf.make_addplot(df_plot["Signal"], panel=macd_panel, color='red'))
        if "MACD_Hist" in df_plot.columns:
            apds.append(mpf.make_addplot(df_plot["MACD_Hist"], panel=macd_panel, type='bar', color='gray', alpha=0.5))
        panel_ratios.append(1)

    # Plot config
    plot_kwargs = dict(
        type='candle',
        style='charles',
        title=title,
        ylabel='Price',
        ylabel_lower='Volume',
        volume=True,
        addplot=apds,
        figscale=1.2,
        figratio=(14, 9),
        panel_ratios=panel_ratios
    )

    if save_path:
        plot_kwargs["savefig"] = save_path

    mpf.plot(ohlc, **plot_kwargs)

    if save_path:
        print(f"Plot saved to {save_path}")
