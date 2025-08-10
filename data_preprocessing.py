import pandas as pd
import numpy as np

# Load dataset
pd.read_csv("data/clean_loan_small.csv")
print("📂 Loaded shape:", df.shape)

# Drop columns that are not useful for modeling
drop_cols = ['emp_title', 'emp_length', 'address']
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Clean categorical values
df['term'] = df['term'].str.extract(r'(\d+)').astype(float)
df['int_rate'] = pd.to_numeric(df['int_rate'].astype(str).str.rstrip('%'), errors='coerce')
df['revol_util'] = pd.to_numeric(df['revol_util'].astype(str).str.rstrip('%'), errors='coerce')


# Drop rows with too many nulls
df.dropna(thresh=0.9*len(df.columns), inplace=True)

# Fill remaining nulls with median (safe)
df.fillna(df.median(numeric_only=True), inplace=True)

# Drop grade (redundant with sub_grade)
if 'grade' in df.columns:
    df.drop('grade', axis=1, inplace=True)

# One-hot encode categoricals
cat_cols = ['sub_grade', 'home_ownership', 'initial_list_status', 'application_type']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Save cleaned data
df.to_csv("data/clean_loan_v2.csv", index=False)
print("✅ Cleaned shape:", df.shape)
