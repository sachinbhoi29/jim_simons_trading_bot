import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from pandas.tseries.offsets import BDay



class MarketRegimeForecaster:
    def __init__(self, regimes_df=None, file_path=None,combined_actual_and_forecast_file_name = None):
        """
        Initializes the Market Regime Forecaster.
        :param file_path: Path to the processed market data CSV file.
        """
        self.file_path = file_path
        self.combined_actual_and_forecast_file_name = combined_actual_and_forecast_file_name
        self.df = regimes_df
        self.transition_matrix = None
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.regime_mapping = {}
        self.reverse_mapping = {}
        print(self.df.columns)

        # Output directories
        self.plot_dir = "plots/"
        self.data_dir = "data/"
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def load_data(self):
        """Loads the processed market data and computes additional technical indicators."""
        if self.df is None or self.df.empty:
            self.df = pd.read_csv(self.file_path, parse_dates=["Date"], index_col="Date")

    def compute_transition_matrix(self):
        """Computes the Markov transition probability matrix from historical regime changes."""
        unique_regimes = self.df["Enhanced_Regime"].dropna().unique()
        self.regime_mapping = {regime: idx for idx, regime in enumerate(unique_regimes)}
        self.reverse_mapping = {v: k for k, v in self.regime_mapping.items()}
        num_regimes = len(unique_regimes)

        self.df["Regime_ID"] = self.df["Enhanced_Regime"].map(self.regime_mapping)

        # Initialize transition matrix
        self.transition_matrix = np.zeros((num_regimes, num_regimes))

        for i in range(1, len(self.df)):
            prev_regime = self.df["Regime_ID"].iloc[i - 1]
            curr_regime = self.df["Regime_ID"].iloc[i]
            if not np.isnan(prev_regime) and not np.isnan(curr_regime):
                self.transition_matrix[int(prev_regime), int(curr_regime)] += 1

        # Normalize to get probabilities
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)

    def train_machine_learning_model(self):
        """Trains a RandomForest model for predicting market regimes."""
        features = ["50EMA", "200EMA", "RSI", "ATR", "MACD", "Signal", "BB_Width", "OBV", "Support", "Resistance"]
        X = self.df[features].dropna()
        y = self.df["Regime_ID"].loc[X.index]

        # Stratified sampling to keep regime balance
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in sss.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train the Random Forest Model
        self.rf_model.fit(X_train, y_train)

    def predict_future_regimes(self, days=15):
        """Predicts the next 'days' market regimes using a hybrid Markov + ML approach."""
        future_predictions = []
        confidence_scores = []

        # Get last known regime
        state = int(self.df["Regime_ID"].dropna().iloc[-1])
        available_classes = self.rf_model.classes_

        for _ in range(days):
            # Get Markov-based probabilities
            next_state_probs = self.transition_matrix[state]

            # Align Markov to ML model's known classes
            markov_probs = np.zeros_like(available_classes, dtype=float)
            for idx, cls in enumerate(available_classes):
                markov_probs[idx] = next_state_probs[cls] if cls < len(next_state_probs) else 0.0
            markov_probs /= markov_probs.sum()  # Normalize

            # ML-based prediction using last 10 days
            last_n_days = self.df[["50EMA", "200EMA", "RSI", "ATR", "MACD", "Signal", "BB_Width", "OBV", "Support", "Resistance"]].iloc[-10:].values
            ml_predictions = self.rf_model.predict_proba(last_n_days)
            avg_ml_probs = np.mean(ml_predictions, axis=0)

            # Hybrid weighting
            markov_weight = max(0.1, 1 - np.max(avg_ml_probs))
            ml_weight = 1 - markov_weight

            # Combine predictions
            combined_probs = (avg_ml_probs * ml_weight) + (markov_probs * markov_weight)

            # Pick top-2 hybrid probabilities
            top2_indices = np.argsort(combined_probs)[-2:]
            top2_probs = combined_probs[top2_indices]
            top2_probs /= top2_probs.sum()

            # Randomly choose one of the top 2
            combined_prediction = np.random.choice(available_classes[top2_indices], p=top2_probs)
            predicted_label = self.reverse_mapping[combined_prediction]

            future_predictions.append(predicted_label)
            confidence_scores.append(np.max(avg_ml_probs) * 100)
            state = combined_prediction

        # Save Forecast Results
        self.forecast_df = pd.DataFrame({"Day": np.arange(1, days + 1), "Predicted Regime": future_predictions, "Confidence (%)": np.round(confidence_scores, 2)})
        # forecast_file = os.path.join(self.data_dir, "market_regime_forecast.csv")
        # forecast_df.to_csv(forecast_file, index=False)
        # print(f"Forecast saved to {forecast_file}")

    def save_transition_plot(self):
        """Saves the transition probability heatmap plot."""
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.transition_matrix, annot=True, cmap="coolwarm",
                    xticklabels=self.regime_mapping.keys(), yticklabels=self.regime_mapping.keys())
        plt.title("Market Regime Transition Probabilities (Markov + ML Hybrid)")
        plt.xlabel("Next Regime")
        plt.ylabel("Current Regime")

        # Save plot
        plot_file = os.path.join(self.plot_dir, "market_regime_transition_heatmap.png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Transition probability plot saved to {plot_file}")


    def save_combined_actual_and_forecast(self):
        """Saves a combined dataframe with actual data and appended forecasted regimes with adjusted dates."""
        
        # Create a copy of original dataframe with 'Forecasted Regime' column
        combined_df = self.df.copy()
        combined_df["Forecasted Regime"] = np.nan

        # Generate forecasted dates (business days only)
        last_date = combined_df.index[-1]
        forecast_dates = []
        next_day = last_date
        while len(forecast_dates) < len(self.forecast_df):
            next_day += BDay(1)  # next business day
            forecast_dates.append(next_day)

        # Create forecast DataFrame indexed by forecasted dates
        forecast_only_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecasted Regime": self.forecast_df["Predicted Regime"].values
        })
        forecast_only_df.set_index("Date", inplace=True)

        # Create empty columns matching original df
        empty_cols = combined_df.columns.difference(["Forecasted Regime"])
        for col in empty_cols:
            forecast_only_df[col] = np.nan
        forecast_only_df = forecast_only_df[combined_df.columns]  # order columns

        # Combine historical + forecast
        final_df = pd.concat([combined_df, forecast_only_df])
        final_df.index.name = "Date"

        # Save the combined DataFrame
        final_path = os.path.join(self.data_dir, self.combined_actual_and_forecast_file_name)
        final_df.to_csv(final_path)
        print(f"Combined actual and forecasted data saved to {final_path}")

        self.combined_actual_and_forecast_df = final_df


    def run_forecast_pipeline(self):
        """Runs the full forecasting pipeline."""
        print("Loading data...")
        self.load_data()

        print("Computing Markov Transition Matrix...")
        self.compute_transition_matrix()

        print("Training ML model...")
        self.train_machine_learning_model()

        print("Predicting future regimes...")
        self.predict_future_regimes()

        print("Saving transition heatmap plot...")
        self.save_transition_plot()

        print("Saving combined actual and forecast data...")
        self.save_combined_actual_and_forecast()

        return self.combined_actual_and_forecast_df


# === Usage Example ===
if __name__ == "__main__":
    forecaster = MarketRegimeForecaster(file_path="NIFTY50_Refined_Bearish_Regime_Detection.csv")
    forecaster.run_forecast_pipeline()
