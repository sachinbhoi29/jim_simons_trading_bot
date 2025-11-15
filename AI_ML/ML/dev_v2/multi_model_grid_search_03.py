import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.metrics import precision_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# ===============================
# ⚙️ Adjustable Parameters
# ===============================
DATA_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/data/normalized_data_for_ml.csv"
MODEL_SAVE_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/models/"
TRADES_SAVE_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v2/data/"

TARGET_THRESHOLD = 0.001       # top 0.1% future return
MIN_TRADES = 1000              # minimum high-confidence trades
PRECISION_FLOOR = 0.80         # minimum acceptable precision

# Base hyperparameters for search
N_ESTIMATORS = [200, 300]
MAX_DEPTH = [4, 5, 6]
LEARNING_RATE = [0.05, 0.1]
SUBSAMPLE = [0.8, 1.0]
COLSAMPLE = [0.8, 1.0]

# ===============================
# 1️⃣ Load data
# ===============================
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Ticker'] = df['Ticker'].astype(str)

# ===============================
# 2️⃣ Define target
# ===============================
df['target_bin'] = (df['future_return'] > TARGET_THRESHOLD).astype(int)
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

pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# ===============================
# 5️⃣ Automatic threshold selection
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
    return 0.5  # fallback

# ===============================
# 6️⃣ Model GridSearch & Training
# ===============================
# --- XGBoost ---
print("\nGridSearch XGBoost...")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=pos_weight,
    random_state=42,
    n_jobs=-1
)

xgb_param_grid = {
    'n_estimators': N_ESTIMATORS,
    'max_depth': MAX_DEPTH,
    'learning_rate': LEARNING_RATE,
    'subsample': SUBSAMPLE,
    'colsample_bytree': COLSAMPLE
}

xgb_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    scoring='precision',
    cv=3,
    verbose=2,
    n_jobs=-1
)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print("Best XGBoost params:", xgb_search.best_params_)

# --- LightGBM ---
print("\nGridSearch LightGBM...")
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_param_grid = {
    'n_estimators': N_ESTIMATORS,
    'max_depth': MAX_DEPTH,
    'learning_rate': LEARNING_RATE,
    'subsample': SUBSAMPLE,
    'colsample_bytree': COLSAMPLE
}

lgb_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=lgb_param_grid,
    scoring='precision',
    cv=3,
    verbose=2,
    n_jobs=-1
)
lgb_search.fit(X_train, y_train)
best_lgb = lgb_search.best_estimator_
print("Best LightGBM params:", lgb_search.best_params_)

# --- CatBoost (manual grid) ---
print("\nGridSearch CatBoost...")
cat_param_grid = {
    'iterations': N_ESTIMATORS,
    'depth': MAX_DEPTH,
    'learning_rate': LEARNING_RATE
}
best_cat_prec = 0
best_cat = None
for params in ParameterGrid(cat_param_grid):
    model = CatBoostClassifier(
        iterations=params['iterations'],
        depth=params['depth'],
        learning_rate=params['learning_rate'],
        auto_class_weights='Balanced',
        verbose=0,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    prec = precision_score(y_test, y_pred)
    if prec > best_cat_prec:
        best_cat_prec = prec
        best_cat = model
print("Best CatBoost precision:", best_cat_prec)

# ===============================
# 7️⃣ Evaluate best models and save high-confidence trades
# ===============================
def evaluate_model(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    best_threshold = select_threshold(y_test, y_prob)
    print(f"\n{name} high-confidence threshold: {best_threshold:.3f}")
    
    y_pred_highconf = (y_prob >= best_threshold).astype(int)
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
    print(high_conf_trades_sorted[['Date','Ticker','TopProb']].head(10))
    
    # Save model
    joblib.dump(model, f"{MODEL_SAVE_PATH}{name}_model_highconf.pkl")
    high_conf_trades_sorted.to_csv(f"{TRADES_SAVE_PATH}{name}_highconf_trades.csv", index=False)

# Evaluate & save
evaluate_model("XGBoost", best_xgb, X_test, y_test)
evaluate_model("LightGBM", best_lgb, X_test, y_test)
evaluate_model("CatBoost", best_cat, X_test, y_test)
