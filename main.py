from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

# Load model and features
model = joblib.load("fraud_model_lgbm.joblib")
feature_names = joblib.load("feature_names.joblib")
THRESHOLD = 0.72

@app.route("/", methods=["GET"])
def home():
    return render_template("form.html")  # HTML file goes in a templates/ folder

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form data
        claim_dict = {key: request.form.get(key) for key in feature_names}        
        # Typecast numeric fields if necessary
        for key in ["Age", "RepNumber", "WeekOfMonth", "Deductible", "DriverRating"]:
            claim_dict[key] = int(claim_dict[key])       
        input_data = [claim_dict.get(feat, 0) for feat in feature_names]
        proba = model.predict_proba([input_data])[0][1]
        is_fraud = int(proba >= THRESHOLD)
        return render_template(
            "result.html", 
            probability=round(proba, 4), 
            is_fraud=is_fraud
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

