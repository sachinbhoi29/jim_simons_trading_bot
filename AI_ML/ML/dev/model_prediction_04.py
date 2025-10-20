import pandas as pd
import numpy as np
import xgboost as xgb

# -------------------------------
# 1️⃣ Load cleaned dataset
# -------------------------------
df = pd.read_csv("C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev/data/clean_df.csv")

# Convert date & ticker
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Ticker'] = df['Ticker'].astype(str)

# -------------------------------
# 2️⃣ Prepare features
# -------------------------------
drop_cols = ['Date', 'Ticker', 'future_return']
features = [col for col in df.columns if col not in drop_cols]
X = df[features]

# -------------------------------
# 3️⃣ Load trained model
# -------------------------------
bst = xgb.Booster()
bst.load_model("C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev/xgb_multibin_model.json")

# -------------------------------
# 4️⃣ Predict probabilities and bins
# -------------------------------
dX = xgb.DMatrix(X)
y_prob_bins = bst.predict(dX)
y_pred_bins = np.argmax(y_prob_bins, axis=1)

# -------------------------------
# 5️⃣ Build DataFrame for Excel
# -------------------------------
compare_df = X.copy()
compare_df['Predicted_Bin'] = y_pred_bins

# Optional: probabilities for all bins
for i in range(y_prob_bins.shape[1]):
    compare_df[f'Prob_Bin_{i}'] = y_prob_bins[:, i]

# Add Ticker, Date, and actual future return
compare_df['Ticker'] = df['Ticker'].values
compare_df['Date'] = df['Date'].values
compare_df['Future_Return'] = df['future_return'].values

# -------------------------------
# 6️⃣ Save to Excel
# -------------------------------
compare_df.to_excel(
    "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev/predictions_comparison.xlsx",
    index=False
)

print("Predictions saved to Excel!")
