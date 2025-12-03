# check for threshold value for training
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# ===============================
# ⚙️ Adjustable Parameters
# ===============================

DATA_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4/data/normalized_data_for_ml.csv"
MODEL_SAVE_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4/models/"
TRADES_SAVE_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v4/data/"

TARGET_THRESHOLD = 0.002       # top 0.2% future return
MIN_TRADES = 1000             # minimum high-confidence trades to allow
PRECISION_FLOOR = 0.80        # minimum acceptable precision for high-confidence trades

# Model hyperparameters
N_ESTIMATORS = 200
MAX_DEPTH = 4
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE = 0.8

# ===============================
# 1️⃣ Load data
# ===============================
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Ticker'] = df['Ticker'].astype(str)

# ===============================
# 2️⃣ Define target
# ===============================
# for long
df['target_bin'] = (df['future_return'] > TARGET_THRESHOLD).astype(int)

#for short
# df['target_bin'] = (df['future_return'] < -TARGET_THRESHOLD).astype(int)


print("Target counts:\n", df['target_bin'].value_counts())



# ===============================
# 3️⃣ Prepare features
# ===============================
exclude = ['Date', 'Ticker', 'future_return', 'target_bin']
features = [col for col in df.columns if col not in exclude]

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
# 5️⃣ Model definitions
# ===============================
pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

models = {
    'XGBoost': xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        scale_pos_weight=pos_weight
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        random_state=42,
        class_weight='balanced',
    verbose=-1,
    ),
    'CatBoost': CatBoostClassifier(
        iterations=N_ESTIMATORS,
        depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        verbose=0,
        random_state=42,
        auto_class_weights='Balanced'
    )
}

# ===============================
# 6️⃣ Automatic threshold selection function
# ===============================
def select_threshold(y_true, y_prob, min_trades=MIN_TRADES, precision_floor=PRECISION_FLOOR):
    thresholds = np.linspace(0.5, 0.95, 50)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        num_trades = y_pred.sum()
        if num_trades >= min_trades:
            prec = precision_score(y_true, y_pred)
            if prec >= precision_floor:
                return t
    return 0.5  # fallback if no threshold satisfies

# ===============================
# 7️⃣ Train, evaluate, and save
# ===============================
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Automatically select threshold
    best_threshold = select_threshold(y_test, y_prob)
    print(f"Using threshold {best_threshold:.3f} for high-confidence positives")

    # Classify high-confidence positives
    y_pred_highconf = (y_prob >= best_threshold).astype(int)

    # Evaluate
    precision = precision_score(y_test, y_pred_highconf)
    cm = confusion_matrix(y_test, y_pred_highconf)
    print(f"{name} High-Confidence Precision: {precision:.4f}")
    print("Confusion Matrix (TP/FP focused):\n", cm)

    # High-confidence trades
    X_test_df = X_test.copy()
    X_test_df['TopProb'] = y_prob
    X_test_df['Ticker'] = df.loc[test_mask, 'Ticker'].values
    X_test_df['Date'] = df.loc[test_mask, 'Date'].values
    high_conf_trades = X_test_df[X_test_df['TopProb'] >= best_threshold]
    high_conf_trades_sorted = high_conf_trades.sort_values(by='TopProb', ascending=False)

    print("Top 10 high-confidence positive trades:")
    print(high_conf_trades_sorted[['Date', 'Ticker', 'TopProb']].head(10))

    # Save model
    model_path = f"{MODEL_SAVE_PATH}{name}_model_highconf.pkl"
    joblib.dump(model, model_path)
    print(f"{name} model saved at {model_path}")

    # Save high-confidence trades
    trades_path = f"{TRADES_SAVE_PATH}{name}_highconf_trades.csv"
    # high_conf_trades_sorted.to_csv(trades_path, index=False)
    # print(f"High-confidence trades saved at {trades_path}")
