from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn


# Load model and feature names
model = joblib.load("fraud_model_lgbm.joblib")
feature_names = joblib.load("feature_names.joblib")

# Threshold for classification
THRESHOLD = 0.72

app = FastAPI(title="Insurance Fraud Detection API")

# Request schema
class InsuranceClaim(BaseModel):
    features: dict

@app.get("/")
def read_root():
    return {"message": "Fraud detection API is running"}
    
@app.post("/predict")
def predict(claim: InsuranceClaim):
    # Extract features from input and arrange in correct order
    input_data = [claim.features.get(feat, 0) for feat in feature_names]
    
    # Predict probability
    proba = model.predict_proba([input_data])[0][1]
    is_fraud = int(proba >= THRESHOLD)
    
    return {
        "fraud_probability": round(proba, 4),
        "is_fraud": is_fraud,
        "threshold": THRESHOLD
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
