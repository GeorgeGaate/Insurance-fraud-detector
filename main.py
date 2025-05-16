from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import joblib

app = FastAPI()

model = joblib.load("fraud_model_lgbm.joblib")
feature_names = joblib.load("feature_names.joblib")
THRESHOLD = 0.72

@app.get("/")
def home():
    return {"message": "Fraud detection API is running"}

@app.post("/predict")
def predict(
    Age: int = Form(...),
    RepNumber: int = Form(...),
    WeekOfMonth: int = Form(...),
    Deductible: int = Form(...),
    DriverRating: int = Form(...),
    BasePolicy: str = Form(...),
    Fault: str = Form(...),
    AddressChange_Claim: str = Form(...),
    VehicleCategory: str = Form(...),
    MonthClaimed: str = Form(...),
    Make: str = Form(...),
    DayOfWeekClaimed: str = Form(...),
    PoliceReportFiled: str = Form(...),
    AgeOfPolicyHolder: str = Form(...),
    NumberOfSuppliments: str = Form(...)
):
    try:
        claim_dict = {
            "Age": Age,
            "RepNumber": RepNumber,
            "WeekOfMonth": WeekOfMonth,
            "Deductible": Deductible,
            "DriverRating": DriverRating,
            "BasePolicy": BasePolicy,
            "Fault": Fault,
            "AddressChange_Claim": AddressChange_Claim,
            "VehicleCategory": VehicleCategory,
            "MonthClaimed": MonthClaimed,
            "Make": Make,
            "DayOfWeekClaimed": DayOfWeekClaimed,
            "PoliceReportFiled": PoliceReportFiled,
            "AgeOfPolicyHolder": AgeOfPolicyHolder,
            "NumberOfSuppliments": NumberOfSuppliments
        }
        input_data = [claim_dict.get(feat, 0) for feat in feature_names]
        proba = model.predict_proba([input_data])[0][1]
        is_fraud = int(proba >= THRESHOLD)
        return {
            "fraud_probability": round(proba, 4),
            "is_fraud": is_fraud,
            "threshold": THRESHOLD
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

