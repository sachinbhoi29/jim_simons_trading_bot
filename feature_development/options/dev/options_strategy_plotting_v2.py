# numbers validated
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

class OptionLeg:
    """Represents a single option leg (Call/Put, Buy/Sell, Strike, Premium, Expiry)."""
    def __init__(self, opt_type, action, strike, premium, expiry=None):
        self.opt_type = opt_type.upper()  # 'C' or 'P'
        self.action = action.upper()      # 'BUY' or 'SELL'
        self.strike = strike
        self.premium = premium
        self.expiry = expiry

    def __repr__(self):
        return f"{self.action} {self.opt_type}{self.strike} @ {abs(self.premium)} (Expiry: {self.expiry})"


class OptionStrategy:
    """Option strategy class supporting multiple expiries."""
    def __init__(self, df, expiry=None):
        """
        df: full dataframe with multiple expiries
        expiry: optional, filter for a particular expiry
        """
        self.df_full = df.copy()
        if expiry:
            self.df = df[df['expiryDate'] == expiry].reset_index(drop=True)
        else:
            # default to first expiry in df
            first_expiry = df['expiryDate'].dropna().iloc[0]
            self.df = df[df['expiryDate'] == first_expiry].reset_index(drop=True)
        
        self.legs = []

        # Safe spot price
        self.spot_price = self._get_valid_spot(self.df)

        # Safe lot size determination
        self.lot_size = self._get_lot_size(self._get_valid_underlying(self.df))

        # Default expiry for strategy (if leg does not specify)
        self.expiry = pd.to_datetime(self.df['expiryDate'].dropna().iloc[0])

    def _get_valid_spot(self, df):
        """Pick first non-null, non-zero underlying value from CE or PE."""
        spot_series = df['CE_underlyingValue'].replace(0, np.nan).dropna() if 'CE_underlyingValue' in df.columns else pd.Series()
        if spot_series.empty:
            spot_series = df['PE_underlyingValue'].replace(0, np.nan).dropna() if 'PE_underlyingValue' in df.columns else pd.Series()
        if spot_series.empty:
            return 0
        return spot_series.iloc[0]

    def _get_valid_underlying(self, df):
        """Pick first non-null underlying name."""
        underlying_series = df['CE_underlying'].dropna() if 'CE_underlying' in df.columns else pd.Series()
        if underlying_series.empty:
            underlying_series = df['PE_underlying'].dropna() if 'PE_underlying' in df.columns else pd.Series()
        if underlying_series.empty:
            return 'NIFTY50'
        return str(underlying_series.iloc[0])

    def _get_lot_size(self, underlying):
        lot_sizes = {
            'NIFTY': 75,
            'NIFTY50': 75,
            'BANKNIFTY': 15,
            'FINNIFTY': 40,
            'MIDCPNIFTY': 75
        }
        return lot_sizes.get(underlying.upper(), 75)

    def add_leg(self, opt_type, action, strike, expiry=None):
        """Add a leg; optionally filter by expiry."""
        df = self.df
        if expiry:
            df = self.df_full[self.df_full['expiryDate'] == expiry]
            if df.empty:
                raise ValueError(f"No data found for expiry {expiry}")

        if opt_type.upper() == 'C':
            premium_series = df.loc[df['CE_strikePrice']==strike, 'CE_lastPrice'].replace(0, np.nan).dropna()
        else:
            premium_series = df.loc[df['PE_strikePrice']==strike, 'PE_lastPrice'].replace(0, np.nan).dropna()

        if premium_series.empty:
            raise ValueError(f"No valid premium found for {opt_type}{strike} on expiry {expiry or self.expiry.strftime('%d-%b-%Y')}")

        premium = premium_series.iloc[0]
        if action.upper() == 'SELL':
            premium = -premium

        leg = OptionLeg(opt_type, action, strike, premium, expiry if expiry else self.expiry.strftime('%d-%b-%Y'))
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
        return net_payoff * self.lot_size  # scale by lot size

    def estimate_margin(self):
        """Estimate option selling margin."""
        total_margin = 0
        base_sell_margin = {
            'NIFTY': 195000,
            'NIFTY50': 195000,
            'BANKNIFTY': 130000,
            'FINNIFTY': 60000,
            'MIDCPNIFTY': 75000
        }

        underlying = self._get_valid_underlying(self.df)
        sell_margin_per_lot = base_sell_margin.get(underlying.upper(), 195000)

        has_buy_leg = any(leg.action == 'BUY' for leg in self.legs)
        has_sell_leg = any(leg.action == 'SELL' for leg in self.legs)

        for leg in self.legs:
            if leg.action == 'BUY':
                total_margin += abs(leg.premium) * self.lot_size
            elif leg.action == 'SELL':
                total_margin += sell_margin_per_lot  # fixed per-lot estimate

        # Apply hedging benefit
        if has_buy_leg and has_sell_leg:
            total_margin *= 0.6

        return total_margin

    def plot_payoff(self, price_range, std_dev=1):
        net_payoff = self.calculate_payoff(price_range)
        spot = self.spot_price

        # sigma = 2% of spot by default
        sigma = std_dev * (spot * 0.02)

        zone_1sigma_min = spot - sigma
        zone_1sigma_max = spot + sigma
        zone_2sigma_min = spot - 2*sigma
        zone_2sigma_max = spot + 2*sigma

        x_min = max(price_range.min(), zone_2sigma_min)
        x_max = min(price_range.max(), zone_2sigma_max)

        zoom_mask = (price_range >= x_min) & (price_range <= x_max)
        y_zoom_min = net_payoff[zoom_mask].min()
        y_zoom_max = net_payoff[zoom_mask].max()
        y_padding = (y_zoom_max - y_zoom_min) * 0.1 if y_zoom_max != y_zoom_min else 1000
        y_min = y_zoom_min - y_padding
        y_max = y_zoom_max + y_padding

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=price_range, y=net_payoff,
            mode='lines', name='Payoff',
            line=dict(color='blue'),
            hovertemplate='Price: %{x:.0f}<br>P/L: ₹%{y:,.0f}<extra></extra>'
        ))

        # 2σ zones
        fig.add_shape(type="rect", x0=zone_2sigma_min, y0=y_min, x1=zone_1sigma_min, y1=y_max, fillcolor="lightblue", opacity=0.3, line_width=0)
        fig.add_shape(type="rect", x0=zone_1sigma_max, y0=y_min, x1=zone_2sigma_max, y1=y_max, fillcolor="lightblue", opacity=0.3, line_width=0)

        # 1σ zones
        fig.add_shape(type="rect", x0=zone_1sigma_min, y0=y_min, x1=zone_1sigma_max, y1=y_max, fillcolor="lightgreen", opacity=0.4, line_width=0)

        # Spot price
        fig.add_vline(x=spot, line=dict(color='grey', dash='dot'),
                      annotation_text=f'Spot: {spot:.0f}', annotation_position='top right')

        title_text = f"Payoff at Expiry (Underlying: {self._get_valid_underlying(self.df)}, Lot Size: {self.lot_size})"
        fig.update_layout(
            title=title_text,
            xaxis_title='Underlying Price at Expiry',
            yaxis_title='Profit / Loss',
            xaxis=dict(range=[x_min, x_max], tickformat=',d', tickmode='auto'),
            yaxis=dict(range=[y_min, y_max], zeroline=True, zerolinewidth=2, zerolinecolor='black'),
            hovermode='x unified'
        )

        fig.show()


# ============================
# Example usage
# ============================

# Load CSV
df = pd.read_csv("feature_development/options/dev/NIFTY_optionchain_raw.csv")

# Initialize strategy
strategy = OptionStrategy(df, expiry=None)

# Add legs with different expiries
strategy.add_leg('P', 'SELL', 25650, "30-Sep-2025")
strategy.add_leg('P', 'BUY', 25400, "30-Sep-2025")

# Show strategy
strategy.show_strategy()

# Estimate margin
margin = strategy.estimate_margin()
print(f"Estimated Margin: ₹{margin:,.0f}")

# Plot payoff
price_range = np.arange(25000, 26000, 50)
strategy.plot_payoff(price_range)
