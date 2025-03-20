import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# 🔹 Load the dataset
df = pd.read_csv("stroke_data.csv")

# 🔹 Standardize column names
df.columns = df.columns.str.strip().str.lower()

# 🔹 Handle missing values for 'bmi'
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# 🔹 Convert categorical columns to numeric
df['smoking_status'] = df['smoking_status'].map({'never smoked': 0, 'formerly smoked': 1, 'smokes': 2}).fillna(0)

# 🔹 Derived features
df['hypertension_glucose'] = df['hypertension'] * df['avg_glucose_level']
df['heart_smoking'] = df['heart_disease'] * df['smoking_status']

# 🔹 Select numerical features for scaling (MUST match `main.py`)
numeric_features = ['age', 'bmi', 'avg_glucose_level', 'hypertension_glucose', 'heart_smoking']

# 🔹 Fit the StandardScaler
scaler = StandardScaler()
scaler.fit(df[numeric_features])  # ✅ Fit only on numerical training data

# 🔹 Save the trained scaler
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved as 'scaler.pkl'")
