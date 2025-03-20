from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import traceback

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load trained model and scaler
model = joblib.load("best_stroke_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

expected_features = model.get_booster().feature_names
print("🔹 Model expects features:", expected_features)

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_stroke(
    request: Request,
    age: float = Form(...),
    hypertension: int = Form(...),
    heart_disease: int = Form(...),
    ever_married: int = Form(...),
    avg_glucose_level: float = Form(...),
    bmi: float = Form(...),
    smoking_status: int = Form(...),
    gender: str = Form(...),
    work_type: str = Form(...),
    residence_type: str = Form(...),
):
    try:
        # 🔹 Derived Features (Must match `create_scaler.py`)
        hypertension_glucose = hypertension * avg_glucose_level
        heart_smoking = heart_disease * smoking_status

        # 🔹 One-Hot Encoding for Categorical Features
        gender_Male = int(gender == "Male")
        gender_Other = int(gender == "Other")
        work_type_Private = int(work_type == "Private")
        work_type_Self_employed = int(work_type == "Self-employed")
        work_type_children = int(work_type == "children")
        work_type_Never_worked = int(work_type == "Never worked")
        residence_type_Urban = int(residence_type == "Urban")

        # 🔹 BMI Category
        bmi_category_1 = int(18.5 <= bmi < 25)
        bmi_category_2 = int(25 <= bmi < 30)
        bmi_category_3 = int(bmi >= 30)

        # 🔹 Age Group
        age_group_1 = int(age < 30)
        age_group_2 = int(30 <= age < 50)
        age_group_3 = int(age >= 50)

        # 🔹 Create Feature DataFrame (Ensure the order matches `create_scaler.py`)
        numerical_features = pd.DataFrame([[
            age, bmi, avg_glucose_level, hypertension_glucose, heart_smoking
        ]], columns=['age', 'bmi', 'avg_glucose_level', 'hypertension_glucose', 'heart_smoking'])

        # 🔹 Scale Numerical Features
        numerical_features_scaled = scaler.transform(numerical_features)

        # 🔹 Create Categorical Feature Vector
        categorical_features = np.array([[
            hypertension, heart_disease, ever_married, smoking_status,
            gender_Male, gender_Other, work_type_Never_worked, work_type_Private,
            work_type_Self_employed, work_type_children, residence_type_Urban,
            bmi_category_1, bmi_category_2, bmi_category_3, age_group_1, age_group_2, age_group_3
        ]]).reshape(1, -1)  # ✅ Ensures it is 2D

        # 🔹 Combine Numerical & Categorical Features
        input_features = np.hstack((numerical_features_scaled, categorical_features))

        # 🔹 Make Prediction
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]  # Stroke probability

        # 🔹 Debugging Logs
        print("\n🔍 Final Model Input Features:", input_features)
        print(f"🔍 Model Prediction: {prediction}")
        print(f"🔍 Stroke Probability: {probability:.4f}")

        # 🔹 Risk Assessment
        if probability > 0.3:
            result_message = "⚠️ High Risk: Consult a Doctor!"
        else:
            result_message = "✅ Low Risk: Stay Healthy!"

        return templates.TemplateResponse("index.html", {"request": request, "result": result_message})

    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        print("\n🔴 FULL ERROR TRACEBACK 🔴\n", traceback.format_exc())

        return templates.TemplateResponse("index.html", {"request": request, "result": error_message})
