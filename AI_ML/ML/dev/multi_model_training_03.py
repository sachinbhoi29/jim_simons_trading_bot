import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 1️⃣ Load and preprocess data
# ===============================
df = pd.read_csv("C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev/data/clean_df.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Ticker'] = df['Ticker'].astype(str)

# ===============================
# 2️⃣ Binary target (0.2% 5-day return)
# ===============================
threshold = 0.2
df['target_bin'] = (df['future_return'] > threshold).astype(int)
print("Target counts:\n", df['target_bin'].value_counts())

# ===============================
# 3️⃣ Feature engineering
# ===============================

# EMA, MACD, ATR differences
if 'EMA_20_rel' in df.columns and 'EMA_50_rel' in df.columns:
    df['EMA_diff'] = df['EMA_20_rel'] - df['EMA_50_rel']

if 'MACD_z' in df.columns and 'MACD_signal_z' in df.columns:
    df['MACD_diff'] = df['MACD_z'] - df['MACD_signal_z']

if 'ATR_rel' in df.columns and 'Range_pct' in df.columns:
    df['ATR_Range_ratio'] = df['ATR_rel'] / (df['Range_pct'] + 1e-6)

if 'RSI_14_z' in df.columns and 'NIFTY_RSI_14_z' in df.columns:
    df['RSI_diff'] = df['RSI_14_z'] - df['NIFTY_RSI_14_z']

if 'VWAP_rel' in df.columns and 'NIFTY_EMA_20_rel' in df.columns:
    df['VWAP_Nifty_diff'] = df['VWAP_rel'] - df['NIFTY_EMA_20_rel']

# Optional: add lagged returns for momentum info
lag_cols = ['future_return', 'Range_pct']
for col in lag_cols:
    for lag in [1,2,3]:
        df[f'{col}_lag{lag}'] = df.groupby('Ticker')[col].shift(lag)

df.dropna(inplace=True)  # drop NaNs from lagged features

# ===============================
# 4️⃣ Prepare features
# ===============================
drop_cols = ['Date', 'Ticker', 'future_return', 'target_bin']
features = [col for col in df.columns if col not in drop_cols]

X = df[features]
y = df['target_bin']

# ===============================
# 5️⃣ Train/test split by ticker
# ===============================
tickers = df['Ticker'].unique()
train_tickers, test_tickers = train_test_split(tickers, test_size=0.2, random_state=42)
train_mask = df['Ticker'].isin(train_tickers)
test_mask = df['Ticker'].isin(test_tickers)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# ===============================
# 6️⃣ Define models with imbalance handling
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
# 7️⃣ Train, evaluate, rank
# ===============================
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Tune threshold by probability ranking (optional)
    threshold = np.percentile(y_prob, 90)  # top 10% predictions
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{name} Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print("Confusion Matrix:\n", cm)
    
    # High-confidence positives ranking
    X_test_df = X_test.copy()
    X_test_df['TopProb'] = y_prob
    X_test_df['Ticker'] = df.loc[test_mask, 'Ticker'].values
    X_test_df['Date'] = df.loc[test_mask, 'Date'].values
    
    ranking = X_test_df.sort_values(by='TopProb', ascending=False)
    print("Top 10 high-confidence positives:")
    print(ranking[['Date', 'Ticker', 'TopProb']].head(10))
    
    # Save models
    model_path = f"C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev/{name}_model_highconf.pkl"
    joblib.dump(model, model_path)
    print(f"{name} model saved at {model_path}")
