import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# 🔹 Load the dataset
df = pd.read_csv("stroke_data.csv")

# 🔹 Standardize column names
df.columns = df.columns.str.strip().str.lower()

# 🔹 Handle missing values for 'bmi'
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# 🔹 Select numerical features for scaling (MUST match model training!)
numeric_features = ['age', 'bmi', 'avg_glucose_level', 'hypertension']

# 🔹 Fit the StandardScaler
scaler = StandardScaler()
scaler.fit(df[numeric_features])  # ✅ Fit only on training data

# 🔹 Save the trained scaler
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved as 'scaler.pkl'")
