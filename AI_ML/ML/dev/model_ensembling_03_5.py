import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# ===============================
# 1️⃣ Load test data
# ===============================
df_test = pd.read_csv("C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev/data/clean_df.csv")
df_test['Date'] = pd.to_datetime(df_test['Date'], dayfirst=True, errors='coerce')
df_test['Ticker'] = df_test['Ticker'].astype(str)

# Binary target (if available)
threshold = 0.2
df_test['target_bin'] = (df_test['future_return'] > threshold).astype(int) if 'future_return' in df_test.columns else None

# Feature engineering (same as training!)
if 'EMA_20_rel' in df_test.columns and 'EMA_50_rel' in df_test.columns:
    df_test['EMA_diff'] = df_test['EMA_20_rel'] - df_test['EMA_50_rel']

if 'MACD_z' in df_test.columns and 'MACD_signal_z' in df_test.columns:
    df_test['MACD_diff'] = df_test['MACD_z'] - df_test['MACD_signal_z']

if 'ATR_rel' in df_test.columns and 'Range_pct' in df_test.columns:
    df_test['ATR_Range_ratio'] = df_test['ATR_rel'] / (df_test['Range_pct'] + 1e-6)

if 'RSI_14_z' in df_test.columns and 'NIFTY_RSI_14_z' in df_test.columns:
    df_test['RSI_diff'] = df_test['RSI_14_z'] - df_test['NIFTY_RSI_14_z']

if 'VWAP_rel' in df_test.columns and 'NIFTY_EMA_20_rel' in df_test.columns:
    df_test['VWAP_Nifty_diff'] = df_test['VWAP_rel'] - df_test['NIFTY_EMA_20_rel']

# Prepare features
drop_cols = ['Date', 'Ticker', 'future_return', 'target_bin']
features = [col for col in df_test.columns if col not in drop_cols]
X_test = df_test[features]

# ===============================
# 2️⃣ Load models
# ===============================
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev/XGBoost_model_highconf.json")

lgb_model = joblib.load("C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev/LightGBM_model_highconf.pkl")

cat_model = CatBoostClassifier()
cat_model.load_model("C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev/CatBoost_model_highconf.cbm")

# ===============================
# 3️⃣ Predict probabilities
# ===============================
prob_xgb = xgb_model.predict_proba(X_test)[:,1]
prob_lgb = lgb_model.predict_proba(X_test)[:,1]
prob_cat = cat_model.predict_proba(X_test)[:,1]

# ===============================
# 4️⃣ Ensemble (weighted average)
# ===============================
# You can adjust weights depending on model performance
ensemble_prob = 0.4*prob_xgb + 0.3*prob_lgb + 0.3*prob_cat

# Binarize if needed
threshold_bin = 0.5  # or tune for top X% picks
ensemble_pred = (ensemble_prob >= threshold_bin).astype(int)

# ===============================
# 5️⃣ Rank high-confidence positives
# ===============================
df_result = df_test[['Date','Ticker']].copy()
df_result['EnsembleProb'] = ensemble_prob
df_result['EnsemblePred'] = ensemble_pred

top_picks = df_result.sort_values(by='EnsembleProb', ascending=False).head(20)
print("Top 20 high-confidence ensemble picks:")
print(top_picks)
