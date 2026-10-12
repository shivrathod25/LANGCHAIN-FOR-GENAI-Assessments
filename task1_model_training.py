"""
Task 1: Save & Load a Delivery Time Predictor
---------------------------------------------
Module 13: Model Serialisation and Reload
Assessment Code: M16-A1 (Section B - Task 1)

Requirements:
- Generate dummy dataset (100+ rows) with features: distance_km, num_items, rain_flag
  and target: delivery_time_min.
- 80/20 train/test split.
- Train LinearRegression model and print RMSE on test split.
- Save trained model to delivery_model.joblib and print file size in KB.
- Reload model, run prediction on a sample, assert reloaded matches original, and print PASS/FAIL.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_squared_error


def generate_delivery_data(num_samples: int = 150, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic delivery data with realistic patterns."""
    np.random.seed(random_state)
    distance_km = np.round(np.random.uniform(1.0, 18.0, size=num_samples), 2)
    num_items = np.random.randint(1, 10, size=num_samples)
    rain_flag = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])

    # Realistic delivery time calculation formula:
    # Base prep time 10 min + 2.5 min per km + 1.8 min per item + 8 min if raining + noise
    noise = np.random.normal(0, 2.5, size=num_samples)
    delivery_time_min = np.round(
        10.0 + (2.5 * distance_km) + (1.8 * num_items) + (8.0 * rain_flag) + noise,
        1
    )
    # Ensure all delivery times are positive and realistic minimum
    delivery_time_min = np.clip(delivery_time_min, 12.0, 90.0)

    df = pd.DataFrame({
        "distance_km": distance_km,
        "num_items": num_items,
        "rain_flag": rain_flag,
        "delivery_time_min": delivery_time_min
    })
    return df


def calculate_rmse(y_true, y_pred) -> float:
    """Calculate Root Mean Squared Error across scikit-learn versions."""
    try:
        return float(root_mean_squared_error(y_true, y_pred))
    except (NameError, AttributeError):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    print("=" * 65)
    print("TASK 1: Train, Evaluate, Serialise & Reload Delivery Predictor")
    print("=" * 65)

    # 1. Generate dataset (150 rows > 100 rows minimum requirement)
    df = generate_delivery_data(num_samples=150, random_state=42)
    print(f"[+] Generated dummy dataset with {len(df)} rows.")
    print("Sample Records:")
    print(df.head(3).to_string(index=False))

    # Features and Target
    feature_cols = ["distance_km", "num_items", "rain_flag"]
    X = df[feature_cols]
    y = df["delivery_time_min"]

    # 2. Train/Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[+] Data split into 80/20: Train set = {len(X_train)} rows, Test set = {len(X_test)} rows.")

    # 3. Train LinearRegression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate RMSE on test set
    y_pred_test = model.predict(X_test)
    test_rmse = calculate_rmse(y_test, y_pred_test)
    print(f"[+] Model Training Complete.")
    print(f"    LinearRegression Coefficients: {dict(zip(feature_cols, np.round(model.coef_, 3)))}")
    print(f"    Intercept: {model.intercept_:.3f}")
    print(f"    >> Test Split RMSE: {test_rmse:.3f} minutes <<")

    # 4. Save trained model to delivery_model.joblib
    model_filename = "delivery_model.joblib"
    joblib.dump(model, model_filename)
    file_size_kb = os.path.getsize(model_filename) / 1024.0
    print(f"\n[+] Model saved successfully to '{model_filename}'.")
    print(f"    File Size: {file_size_kb:.2f} KB")

    # 5. Reload model from disk and verify predictions
    print("\n[+] Reloading model from disk to test serialisation consistency...")
    reloaded_model = joblib.load(model_filename)

    # Sample test point: 5.5 km, 3 items, rainy (1)
    sample_input = pd.DataFrame([{
        "distance_km": 5.5,
        "num_items": 3,
        "rain_flag": 1
    }])

    orig_pred = model.predict(sample_input)[0]
    reloaded_pred = reloaded_model.predict(sample_input)[0]

    print(f"    Test Sample Input: {sample_input.to_dict(orient='records')[0]}")
    print(f"    Original Model Prediction : {orig_pred:.4f} min")
    print(f"    Reloaded Model Prediction : {reloaded_pred:.4f} min")

    # Verification Assertion
    try:
        assert np.isclose(orig_pred, reloaded_pred, atol=1e-7), (
            f"Prediction mismatch: {orig_pred} != {reloaded_pred}"
        )
        print("\n" + "=" * 25 + " TEST RESULT: PASS " + "=" * 25)
        print("Verification Assertion Passed: Saved and reloaded model outputs are IDENTICAL.")
    except AssertionError as err:
        print("\n" + "=" * 25 + " TEST RESULT: FAIL " + "=" * 25)
        print(f"Verification Assertion Failed: {err}")


if __name__ == "__main__":
    main()
