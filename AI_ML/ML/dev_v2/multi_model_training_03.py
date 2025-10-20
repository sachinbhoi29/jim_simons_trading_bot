import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 1️⃣ Load ML-ready data
# ===============================
df = pd.read_csv("C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/data/normalized_data_for_ml.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Ticker'] = df['Ticker'].astype(str)

# ===============================
# 2️⃣ Define binary target (top 0.2% future return)
# ===============================
threshold = 0.002
df['target_bin'] = (df['future_return'] > threshold).astype(int)
print("Target counts:\n", df['target_bin'].value_counts())

# ===============================
# 3️⃣ Prepare features
# ===============================
filtered_features = [
    # EMA features
    'EMA_diff_rel', 'EMA_20_rel', 'EMA_50_rel', 'EMA_fast_rel', 'EMA_slow_rel',
    'Close_vs_EMA20', 'Close_vs_EMA50',
    
    # Bollinger
    'BB_Width', 'BB_Pos', 'BB_Mid_Dev',
    
    # ATR / Range
    'ATR_Range_ratio', 'Range_over_ATR', 'Range_over_Close',
    
    # Momentum
    'MACD_diff', 'RSI_diff_NIFTY', 'RSI_diff_BANKNIFTY', 'RSI_vs_MACD',
    
    # VWAP / Volume
    'VWAP_rel', 'VWAP_minus_Close', 'VWAP_minus_EMA20',
    'Volume_log', 'Cum_Vol_log', 'Cum_TPV_log',
    
    # Regime
    'NIFTY_Enhanced_Regime_for_RD', 'BANKNIFTY_Enhanced_Regime_for_RD'
]

features = filtered_features

X = df[features]
y = df['target_bin']

# ===============================
# 4️⃣ Train/test split by ticker
# ===============================
tickers = df['Ticker'].unique()
train_tickers, test_tickers = train_test_split(tickers, test_size=0.2, random_state=42)
train_mask = df['Ticker'].isin(train_tickers)
test_mask = df['Ticker'].isin(test_tickers)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# ===============================
# 5️⃣ Model definitions (imbalanced)
# ===============================
pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

models = {
    'XGBoost': xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight='balanced'
    ),
    'CatBoost': CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        verbose=0,
        random_state=42,
        auto_class_weights='Balanced'
    )
}

# ===============================
# 6️⃣ Train, evaluate, and save
# ===============================
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{name} Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print("Confusion Matrix:\n", cm)
    
    # High-confidence ranking
    X_test_df = X_test.copy()
    X_test_df['TopProb'] = y_prob
    X_test_df['Ticker'] = df.loc[test_mask, 'Ticker'].values
    X_test_df['Date'] = df.loc[test_mask, 'Date'].values
    
    ranking = X_test_df.sort_values(by='TopProb', ascending=False)
    print("Top 10 high-confidence positives:")
    print(ranking[['Date', 'Ticker', 'TopProb']].head(10))
    
    # Save model
    model_path = f"C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/models/{name}_model_highconf.pkl"
    joblib.dump(model, model_path)
    print(f"{name} model saved at {model_path}")
