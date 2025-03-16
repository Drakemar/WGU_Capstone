import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load dataset
df = pd.read_csv("stroke_data.csv")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# 🔹 1️⃣ Feature Importance Plot
model = joblib.load("best_stroke_prediction_model.pkl")
feature_importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10))
plt.title("Top 10 Important Features for Stroke Prediction")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.savefig("static/feature_importance.png")
plt.close()

# 🔹 2️⃣ Stroke Risk by Age Histogram
plt.figure(figsize=(10, 5))
sns.histplot(df[df["stroke"] == 1]["age"], bins=20, kde=True, color="red", label="Stroke Patients")
sns.histplot(df[df["stroke"] == 0]["age"], bins=20, kde=True, color="blue", label="No Stroke")
plt.legend()
plt.title("Stroke Occurrence by Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("static/stroke_by_age.png")
plt.close()

# 🔹 3️⃣ BMI vs Glucose Scatter Plot
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["bmi"], y=df["avg_glucose_level"], hue=df["stroke"], alpha=0.7)
plt.title("BMI vs Glucose Level (Colored by Stroke)")
plt.xlabel("BMI")
plt.ylabel("Average Glucose Level")
plt.savefig("static/bmi_vs_glucose.png")
plt.close()

print("✅ Visualizations saved in 'static/' folder!")
