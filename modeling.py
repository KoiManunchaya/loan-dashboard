import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ================= Load & Clean =================
df = pd.read_csv('data/clean_loan_v2.csv')
print("📂 Loaded df shape:", df.shape)
print(df.head())

# 🔥 Drop columns we won't use
df.drop(columns=['id', 'member_id'], inplace=True, errors='ignore')

# 🔍 Show NaN summary
nan_summary = df.isnull().sum()
nan_cols = nan_summary[nan_summary > 0]
print(f"🔍 Columns with NaN:\n{nan_cols}")

# 🧼 Drop columns with >50% NaN
to_drop = nan_cols[nan_cols > 0.5 * len(df)].index.tolist()
df.drop(columns=to_drop, inplace=True)

# ✅ Fill remaining NaN with median (safe for numeric)
df.fillna(df.median(numeric_only=True), inplace=True)

# ✅ Ensure no missing
print("✅ Final NaN check:", df.isnull().sum().sum())

# ================= Prepare X and y =================
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# ✅ Encode y target
y = y.map({'Fully Paid': 0, 'Charged Off': 1})

# ✅ Drop object columns (XGBoost doesn’t support string)
X = X.drop(columns=X.select_dtypes(include='object').columns)

# ✅ Convert category columns to numeric codes
for col in X.select_dtypes(include='category').columns:
    X[col] = X[col].cat.codes

# ✅ Confirm valid dtypes
print("✅ X dtypes after cleaning:", X.dtypes.unique())

# ================= Scale and Split =================
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

# ================= Modeling =================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# ================= Evaluation =================
models = {
    "Logistic Regression": log_preds,
    "Random Forest": rf_preds,
    "XGBoost": xgb_preds
}

for name, preds in models.items():
    print(f"\n📊 Model: {name}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

# ================= Plot Confusion Matrix =================
plt.figure(figsize=(12, 4))
for i, (name, preds) in enumerate(models.items(), 1):
    plt.subplot(1, 3, i)
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ================= Export Confusion Matrix Table =================
confusion_results = []
for name, preds in models.items():
    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm, columns=["Predicted: Fully Paid", "Predicted: Charged Off"],
                         index=["Actual: Fully Paid", "Actual: Charged Off"])
    cm_df.insert(0, 'Model', name)
    confusion_results.append(cm_df.reset_index())

final_cm_table = pd.concat(confusion_results, axis=0).rename(columns={"index": "Actual"})
print("\n📥 Combined Confusion Matrix Table:")
print(final_cm_table)
final_cm_table.to_csv('data/confusion_matrix_results.csv', index=False)

# ================= Feature Importance (RF) =================
importance_df = pd.DataFrame({
    'Feature': numeric_cols,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n🔥 Top 10 Feature Importance from RF:")
print(importance_df.head(10))
importance_df.head(10).to_csv('data/feature_importance.csv', index=False)

# ================= Class Balance Check =================
print("\n📊 Class Proportion:")
print(y.value_counts(normalize=True))
