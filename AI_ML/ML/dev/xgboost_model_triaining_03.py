import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb

# 1️⃣ Load preprocessed data
df = pd.read_csv("C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev/data/clean_df.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Ticker'] = df['Ticker'].astype(str)

# 2️⃣ Binary target
# threshold = 0.5
threshold = 0.2
df['target_bin'] = (df['future_return'] > threshold).astype(int)
print("Target counts:\n", df['target_bin'].value_counts())

# 3️⃣ Features
drop_cols = ['Date', 'Ticker', 'future_return', 'target_bin']
features = [col for col in df.columns if col not in drop_cols]

X = df[features]
y = df['target_bin']

# 4️⃣ Train/test split by ticker
tickers = df['Ticker'].unique()
train_tickers, test_tickers = train_test_split(tickers, test_size=0.2, random_state=42)
train_mask = df['Ticker'].isin(train_tickers)
test_mask = df['Ticker'].isin(test_tickers)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# 5️⃣ XGBoost classifier with hyperparameter tuning
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)

param_grid = {
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
}

grid = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='accuracy',  # metric not too important for ranking
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# 6️⃣ Predict probabilities
y_prob = best_model.predict_proba(X_test)[:, 1]  # probability of positive
y_pred = best_model.predict(X_test)

# Optional evaluation
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Binary Classification Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)

# 7️⃣ Rank by probability (high-confidence positives)
X_test_df = X_test.copy()
X_test_df['TopProb'] = y_prob
X_test_df['Ticker'] = df.loc[test_mask, 'Ticker'].values
X_test_df['Date'] = df.loc[test_mask, 'Date'].values

# Indices of False Negatives
fn_mask = (y_test == 1) & (y_pred == 0)

# Predicted probabilities for the positive class (class 1)
fn_probs = y_prob[fn_mask]

print("Number of False Negatives:", fn_mask.sum())
print("Predicted probabilities for False Negatives:")
print(fn_probs)


ranking = X_test_df.sort_values(by='TopProb', ascending=False)
print("Top 20 high-confidence positives:")
print(ranking[['Date', 'Ticker', 'TopProb']].head(20))

# 8️⃣ Save model
best_model.save_model("C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev/xgb_binary_model_highconf.json")
print("Binary XGBoost model (high-confidence positives) saved successfully!")
