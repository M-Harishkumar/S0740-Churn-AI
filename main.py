import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pickle
import json

print("🔥 S0740: Customer Churn Prediction - Starting...")

# Load data
df = pd.read_csv('churn_data.csv')
print(f"📊 Data loaded: {df.shape[0]} customers, {df.shape[1]} features")

# Find churn column
churn_column = next(col for col in df.columns if 'churn' in col.lower())
print(f"🎯 Target: {churn_column}")

# Clean data
df = df.dropna()
df[churn_column] = (df[churn_column] == 'Yes').astype(int)

# Encode categorical
text_cols = df.select_dtypes(include=['object']).columns
le_dict = {}
for col in text_cols:
    if col != churn_column:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

# Prepare X, y
X = df.drop(columns=[churn_column])
y = df[churn_column]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Save preprocessing
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(le_dict, open('encoders.pkl', 'wb'))
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("✅ Data cleaning complete!")

# Train 3 models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
}

results = {}
for name, model in models.items():
    print(f"🔧 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    print(f"  ✅ Accuracy: {results[name]['accuracy']:.3f}")

# Select best model
best_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_name]['model']
pickle.dump(best_model, open('best_model.pkl', 'wb'))

# Save results
results_summary = {k: {sk: float(sv) for sk, sv in v.items() if sk != 'model'} for k, v in results.items()}
results_summary['best_model'] = best_name
with open('results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n🏆 BEST MODEL: {best_name} ({results[best_name]['accuracy']:.3f} accuracy)")
print("✅ TRAINING COMPLETE! Run 'streamlit run dashboard.py' next.")
