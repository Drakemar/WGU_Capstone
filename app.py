from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Ensure a directory for saving plots
os.makedirs("static", exist_ok=True)

@app.get("/visualizations/{plot_name}")
def get_visualization(plot_name: str):
    """Serve visualization images."""
    file_path = f"static/{plot_name}.png"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "Plot not found"}

# 🔹 Load trained model and scaler
model = joblib.load("best_stroke_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")  # ✅ Load the trained scaler used during training
expected_features = model.get_booster().feature_names

print("🔹 Model expects features:", expected_features)

# ✅ Define FastAPI routes
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict_stroke(
    request: Request,
    age: float = Form(),
    hypertension: int = Form(),
    heart_disease: int = Form(),
    ever_married: int = Form(),
    avg_glucose_level: float = Form(),
    bmi: float = Form(),
    smoking_status: int = Form(),
    gender: str = Form(),
    work_type: str = Form(),
    residence_type: str = Form(),
):
    """Process user input, transform it, and make stroke prediction."""

    # 🔹 Derived features
    hypertension_glucose = hypertension * avg_glucose_level
    heart_smoking = heart_disease * smoking_status

    # 🔹 One-hot encoding for categorical features
    gender_Male = int(gender == "Male")
    gender_Other = int(gender == "Other")
    work_type_Private = int(work_type == "Private")
    work_type_Self_employed = int(work_type == "Self-employed")  # ✅ FIXED
    work_type_children = int(work_type == "children")
    work_type_Never_worked = int(work_type == "Never worked")
    residence_type_Urban = int(residence_type == "Urban")

    # 🔹 BMI category
    bmi_category_1 = int(18.5 <= bmi < 25)
    bmi_category_2 = int(25 <= bmi < 30)
    bmi_category_3 = int(bmi >= 30)

    # 🔹 Age group
    age_group_1 = int(age < 30)
    age_group_2 = int(30 <= age < 50)
    age_group_3 = int(age >= 50)

    # 🔹 Feature vector
    input_features = np.array([
        age, hypertension, heart_disease, ever_married, avg_glucose_level, bmi, smoking_status,
        hypertension_glucose, heart_smoking, gender_Male, gender_Other, work_type_Never_worked,
        work_type_Private, work_type_Self_employed, work_type_children, residence_type_Urban,
        bmi_category_1, bmi_category_2, bmi_category_3, age_group_1, age_group_2, age_group_3
    ]).reshape(1, -1)

    print(f"\n🔍 BEFORE SCALING:")
    print(f"→ Age: {age}, BMI: {bmi}, Glucose: {avg_glucose_level}")
    print(f"→ Hypertension Glucose: {hypertension_glucose}, Heart Smoking: {heart_smoking}")

    # 🔹 Apply pre-trained scaler from training
    input_features[:, [0, 4, 5, 7]] = scaler.transform(input_features[:, [0, 4, 5, 7]])  # ✅ Use the same scaler

    print(f"\n🔍 AFTER SCALING:")
    print(f"→ Scaled Age: {input_features[0][0]}, Scaled BMI: {input_features[0][5]}, Scaled Glucose: {input_features[0][4]}")

    # 🔹 Make prediction
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]  # Stroke probability

    # 🔹 Debugging: Print transformed inputs
    print(f"\n🔍 RAW MODEL OUTPUT:")
    print(f"→ Model Predicted Class: {prediction}")
    print(f"→ Stroke Probability: {probability:.4f}")

    # 🔹 Risk Assessment (adjustable threshold)
    threshold = 0.3  # ✅ Lowered for better sensitivity
    if probability > threshold:
        result_message = "⚠️ High Risk: Consult a Doctor!"
    else:
        result_message = "✅ Low Risk: Stay Healthy!"

    return templates.TemplateResponse("index.html", {"request": request, "result": result_message})
