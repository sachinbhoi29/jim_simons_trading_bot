import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load data
# -------------------------------
file_path = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev/data/clean_df.csv"  # replace with your file
df = pd.read_csv(file_path)

# -------------------------------
# Identify numeric and categorical columns
# -------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numeric columns: {numeric_cols}")
print(f"Categorical columns: {cat_cols}")

# -------------------------------
# Summary statistics
# -------------------------------
summary_stats = df[numeric_cols].describe().T
summary_stats['median'] = df[numeric_cols].median()
summary_stats['missing'] = df[numeric_cols].isna().sum()
summary_stats['unique'] = df[numeric_cols].nunique()
print("\nSummary statistics:\n", summary_stats)

# -------------------------------
# Outlier detection (IQR method)
# -------------------------------
outlier_info = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_info[col] = len(outliers)
    
outlier_summary = pd.DataFrame.from_dict(outlier_info, orient='index', columns=['outlier_count'])
print("\nOutlier counts per column:\n", outlier_summary.sort_values('outlier_count', ascending=False))

# -------------------------------
# Plot distributions
# -------------------------------
for col in numeric_cols:
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Histogram of {col}')
    
    plt.subplot(1,2,2)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    
    plt.show()

# -------------------------------
# Optional: Clip extreme outliers
# -------------------------------
# Example: clip at 1st and 99th percentile
clip_percentiles = [0.01, 0.99]
for col in numeric_cols:
    lower, upper = df[col].quantile(clip_percentiles).values
    df[col + "_clipped"] = df[col].clip(lower, upper)

print("\nData sanity check completed. Clipped numeric columns created with suffix '_clipped'.")
