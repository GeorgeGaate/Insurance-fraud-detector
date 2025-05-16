from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and features
model = joblib.load("fraud_model_lgbm.joblib")
feature_names = joblib.load("feature_names.joblib")
THRESHOLD = 0.72

@app.route("/", methods=["GET"])
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form data
        form_data = request.form.to_dict()
        # Convert numerics
        numeric_cols = ["Age", "RepNumber", "WeekOfMonth", "Deductible", "DriverRating"]
        for col in numeric_cols:
            form_data[col] = int(form_data[col])
        # Convert to DataFrame
        df_input = pd.DataFrame([form_data])
        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(df_input)
        # Add missing columns from training
        for col in feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        # Ensure same column order
        df_encoded = df_encoded[feature_names]
        # Predict
        proba = model.predict_proba(df_encoded)[0][1]
        is_fraud = int(proba >= THRESHOLD)
        return render_template("result.html", probability=round(proba, 4), is_fraud=is_fraud)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)

