#  validated this file on NIFTY data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math
import plotly.graph_objects as go

class OptionLeg:
    """Represents a single option leg (Call/Put, Buy/Sell, Strike, Premium)."""
    def __init__(self, opt_type, action, strike, premium):
        self.opt_type = opt_type  # 'C' or 'P'
        self.action = action      # 'BUY' or 'SELL'
        self.strike = strike
        self.premium = premium
    def __repr__(self):
        return f"{self.action} {self.opt_type}{self.strike} @ {abs(self.premium)}"

class OptionStrategy:
    """Main strategy builder."""
    def __init__(self, df):
        self.df = df
        self.legs = []
        self.spot_price = df['spotPrice'].iloc[0]
        self.expiry = pd.to_datetime(df['expiryDate'].iloc[0])
        self.days_to_expiry = max((self.expiry - datetime.today()).days, 1)
        self.lot_size = 0
        underlying = df['CE_underlying'].iloc[0] if 'CE_underlying' in df.columns else 'NIFTY50'
        self.underlying = underlying.upper()
        self.lot_size = self._get_lot_size(self.underlying)
        self.expiry = pd.to_datetime(df['expiryDate'].iloc[0])    
    def _get_lot_size(self, underlying):
        lot_sizes = {
            'NIFTY': 75,
            'NIFTY50': 75,
            'BANKNIFTY': 15,
            'FINNIFTY': 40,
            'MIDCPNIFTY': 75
        }
        return lot_sizes.get(underlying.upper(), 75)  # default to 75 if unknown
    
    def add_leg(self, opt_type, action, strike):
        """Add a leg to the strategy (fetches premium from dataframe)."""
        if opt_type == 'C':
            premium = self.df.loc[self.df['StrikePrice']==strike, 'Call_LTP'].values[0]
        else:
            premium = self.df.loc[self.df['StrikePrice']==strike, 'Put_LTP'].values[0]
        
        if action == 'SELL':
            premium = -premium
        
        leg = OptionLeg(opt_type, action, strike, premium)
        self.legs.append(leg)

    def show_strategy(self):
        """Print all legs in the strategy."""
        for leg in self.legs:
            print(leg)

    def calculate_payoff(self, price_range):
        net_payoff = np.zeros_like(price_range, dtype=float)
        total_premium = 0

        for leg in self.legs:
            if leg.opt_type == 'C':
                payoff = np.maximum(price_range - leg.strike, 0)
            else:
                payoff = np.maximum(leg.strike - price_range, 0)

            if leg.action == 'SELL':
                payoff = -payoff
                total_premium += abs(leg.premium)
            else:
                total_premium -= abs(leg.premium)

            net_payoff += payoff

        net_payoff += total_premium
        return net_payoff * self.lot_size  # 👈 IMPORTANT!

    def estimate_margin(self):
        """Estimate option selling margin based on rough market conditions."""
        total_margin = 0
        base_sell_margin = {
            'NIFTY': 195000,
            'NIFTY50': 195000,
            'BANKNIFTY': 130000,
            'FINNIFTY': 60000,
            'MIDCPNIFTY': 75000
        }

        sell_margin_per_lot = base_sell_margin.get(self.underlying, 195000)

        has_buy_leg = any(leg.action == 'BUY' for leg in self.legs)
        has_sell_leg = any(leg.action == 'SELL' for leg in self.legs)

        for leg in self.legs:
            if leg.action == 'BUY':
                # Premium * lot size
                total_margin += abs(leg.premium) * self.lot_size
            elif leg.action == 'SELL':
                total_margin += sell_margin_per_lot  # fixed per-lot estimate

        # Apply hedging benefit if applicable
        if has_buy_leg and has_sell_leg:
            total_margin *= 0.6  # 40% margin benefit for hedging

        return total_margin






    def plot_payoff(self, price_range, std_dev=1):
        net_payoff = self.calculate_payoff(price_range)
        spot = self.spot_price

        # Assume sigma (std deviation) of price movement is % of spot or use user input
        sigma = std_dev * (spot * 0.02)  # Example: 2% std deviation

        # Define zones for shading
        zone_1sigma_min = spot - sigma
        zone_1sigma_max = spot + sigma
        zone_2sigma_min = spot - 2 * sigma
        zone_2sigma_max = spot + 2 * sigma

        # Limit x-axis to ±2σ range for zoom effect
        x_min = max(price_range.min(), zone_2sigma_min)
        x_max = min(price_range.max(), zone_2sigma_max)

        # Restrict payoff array to zoom region for y-axis scaling
        zoom_mask = (price_range >= x_min) & (price_range <= x_max)
        y_zoom_min = net_payoff[zoom_mask].min()
        y_zoom_max = net_payoff[zoom_mask].max()
        y_padding = (y_zoom_max - y_zoom_min) * 0.1 if y_zoom_max != y_zoom_min else 1000

        y_min = y_zoom_min - y_padding
        y_max = y_zoom_max + y_padding

        fig = go.Figure()

        # Add payoff line
        fig.add_trace(go.Scatter(
            x=price_range, y=net_payoff,
            mode='lines',
            name='Payoff',
            line=dict(color='blue'),
            hovertemplate='Price: %{x:.0f}<br>P/L: ₹%{y:,.0f}<extra></extra>'
        ))

        # Add shaded zones for 2σ (light blue)
        fig.add_shape(type="rect",
                    x0=zone_2sigma_min, y0=y_min, x1=zone_1sigma_min, y1=y_max,
                    fillcolor="lightblue", opacity=0.3, line_width=0)
        fig.add_shape(type="rect",
                    x0=zone_1sigma_max, y0=y_min, x1=zone_2sigma_max, y1=y_max,
                    fillcolor="lightblue", opacity=0.3, line_width=0)

        # Add shaded zones for 1σ (light green)
        fig.add_shape(type="rect",
                    x0=zone_1sigma_min, y0=y_min, x1=zone_1sigma_max, y1=y_max,
                    fillcolor="lightgreen", opacity=0.4, line_width=0)

        # Add vertical line at spot price
        fig.add_vline(x=spot, line=dict(color='grey', dash='dot'),
              annotation_text=f'Spot: {spot:.0f}', annotation_position='top right')


        title_text = f"Payoff at Expiry ({self.underlying}, Exp: {self.expiry.strftime('%d-%b-%Y')}, Lot Size: {self.lot_size})"
        fig.update_layout(
            title=title_text,
            xaxis_title='Underlying Price at Expiry',
            yaxis_title='Profit / Loss',
            xaxis=dict(
                range=[x_min, x_max],
                tickformat=',d',  # 👈 THIS disables 25k-style formatting
                tickmode='auto'
            ),
            yaxis=dict(
                range=[y_min, y_max],
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black'
            ),
            hovermode='x unified'
        )


        fig.show()

# Load CSV
df = pd.read_csv("feature_development/options/dev/NIFTY_options_07Oct2025.csv")

# Initialize strategy
strategy = OptionStrategy(df)

# Add legs
strategy.add_leg('C', 'SELL', 25450)   # Buy Put 23000
strategy.add_leg('P', 'SELL', 25250)   # Buy Put 23000
strategy.add_leg('C', 'BUY', 25550)   # Buy Call 23000
strategy.add_leg('P', 'BUY', 25150)   # Buy Call 23000

# Show legs
strategy.show_strategy()
margin = strategy.estimate_margin()


# Define price range
price_range = np.arange(20000, 27000, 100)

# Plot payoff
strategy.plot_payoff(price_range=np.arange(20000, 27000, 50), std_dev=1)

