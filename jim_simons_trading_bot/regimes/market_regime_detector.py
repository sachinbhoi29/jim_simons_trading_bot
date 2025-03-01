# regimes/market_regime_detector.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import calculate_indicators

class MarketRegimeDetector:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train_model(self, data):
        """Trains the market regime classifier."""
        features, labels = calculate_indicators(data)
        self.model.fit(features, labels)

    def predict_regime(self):
        """Predicts the current market regime."""
        data = pd.read_csv("data/market_data.csv")
        features, _ = calculate_indicators(data)
        return self.model.predict(features)[-1]
