"""
Task 2: Flask REST API for Order Predictions
--------------------------------------------
Module 13: Model Loading combined with Flask API Design
Assessment Code: M16-A1 (Section B - Task 2)

Requirements:
- Load 'delivery_model.joblib' ONCE at startup (not inside the route handler).
- Expose POST /predict endpoint accepting JSON:
  {
      "distance_km": float,
      "num_items": int,
      "rain_flag": int (0 or 1)
  }
- Return JSON: {"predicted_delivery_time_min": float} rounded to 1 decimal place with HTTP 200.
- Return descriptive JSON error with HTTP 400 if any field is missing or invalid,
  without crashing the server.
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- APPLICATION STARTUP: Load Model Once into Memory ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "delivery_model.joblib")

if not os.path.exists(MODEL_PATH):
    # Fallback to current working directory
    MODEL_PATH = "delivery_model.joblib"

print(f"[*] Starting QuickBite Order Prediction API...")
print(f"[*] Loading model from '{MODEL_PATH}' into memory at startup...")

try:
    delivery_model = joblib.load(MODEL_PATH)
    print("[+] Model loaded successfully and kept in memory for worker lifetime.")
except Exception as e:
    print(f"[!] Critical Error loading model from {MODEL_PATH}: {e}")
    delivery_model = None


@app.route("/", methods=["GET"])
def health_check():
    """Service health and metadata endpoint."""
    return jsonify({
        "service": "QuickBite Delivery Prediction API",
        "status": "online",
        "model_loaded": delivery_model is not None,
        "endpoints": {
            "POST /predict": "Predict delivery time in minutes"
        }
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts:
        {
            "distance_km": float,
            "num_items": int,
            "rain_flag": int (0 or 1)
        }
    Returns:
        HTTP 200: {"predicted_delivery_time_min": float}
        HTTP 400: {"error": "...", "missing_fields": [...]}
    """
    if delivery_model is None:
        return jsonify({
            "error": "Model not loaded on server. Please ensure delivery_model.joblib exists."
        }), 500

    # Verify JSON content-type and body
    if not request.is_json:
        return jsonify({
            "error": "Invalid Content-Type. Request body must be application/json."
        }), 400

    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "error": "Empty or malformed JSON payload. Please provide a JSON object."
        }), 400

    # Required field verification
    required_fields = ["distance_km", "num_items", "rain_flag"]
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields in request payload.",
            "missing_fields": missing_fields,
            "required_fields": required_fields,
            "example_valid_payload": {
                "distance_km": 5.2,
                "num_items": 3,
                "rain_flag": 1
            }
        }), 400

    # Data validation and type conversion
    try:
        distance_km = float(data["distance_km"])
        num_items = int(data["num_items"])
        rain_flag = int(data["rain_flag"])

        if distance_km <= 0:
            return jsonify({"error": "distance_km must be a positive number greater than 0."}), 400
        if num_items <= 0:
            return jsonify({"error": "num_items must be a positive integer greater than 0."}), 400
        if rain_flag not in (0, 1):
            return jsonify({"error": "rain_flag must be 0 (no rain) or 1 (raining)."}), 400

    except (ValueError, TypeError) as conv_err:
        return jsonify({
            "error": f"Invalid data types provided: {str(conv_err)}. distance_km must be numeric, num_items and rain_flag must be integers."
        }), 400

    # Perform prediction using loaded model in memory
    try:
        input_df = pd.DataFrame([{
            "distance_km": distance_km,
            "num_items": num_items,
            "rain_flag": rain_flag
        }])
        raw_prediction = delivery_model.predict(input_df)[0]
        predicted_time_min = round(float(raw_prediction), 1)

        return jsonify({
            "predicted_delivery_time_min": predicted_time_min,
            "status": "success",
            "request_inputs": {
                "distance_km": distance_km,
                "num_items": num_items,
                "rain_flag": rain_flag
            }
        }), 200

    except Exception as pred_err:
        return jsonify({
            "error": f"Inference execution failed: {str(pred_err)}"
        }), 500


if __name__ == "__main__":
    # Run the Flask app on localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=False)
