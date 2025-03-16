from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("best_stroke_prediction_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define request body schema
class PatientData(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    smoking_status: int
    ever_married: int
    gender_male: int
    work_type_private: int
    work_type_self_employed: int
    work_type_children: int
    residence_type_urban: int
    bmi_category_1: int
    bmi_category_2: int
    bmi_category_3: int
    age_group_1: int
    age_group_2: int
    age_group_3: int

@app.get("/")
def home():
    return {"message": "Stroke Prediction API is running!"}

@app.post("/predict")
def predict(data: PatientData):
    # Convert input data to numpy array
    input_features = np.array([
        data.age, data.hypertension, data.heart_disease, data.avg_glucose_level, data.bmi,
        data.smoking_status, data.ever_married, data.gender_male, data.work_type_private,
        data.work_type_self_employed, data.work_type_children, data.residence_type_urban,
        data.bmi_category_1, data.bmi_category_2, data.bmi_category_3, data.age_group_1,
        data.age_group_2, data.age_group_3
    ]).reshape(1, -1)

    # Get prediction
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    # Return result
    return {
        "stroke_prediction": int(prediction),
        "probability": round(probability * 100, 2)  # Convert to percentage
    }
