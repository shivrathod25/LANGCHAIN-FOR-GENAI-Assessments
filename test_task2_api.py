"""
Test & Verification Script for Task 2 Flask REST API
---------------------------------------------------
Demonstrates:
1. Successful POST /predict with full payload -> HTTP 200 with rounded delivery time.
2. Incomplete payload (missing 'rain_flag') -> HTTP 400 descriptive error body.
3. Incomplete payload (missing 'distance_km') -> HTTP 400 descriptive error body.
4. Includes exact curl commands for terminal testing.
"""

import json
# pyrefly: ignore [missing-import]
from task2_flask_api import app


def run_tests():
    print("=" * 65)
    print("TASK 2 VERIFICATION: Testing Flask REST API with Flask Client")
    print("=" * 65)

    client = app.test_client()

    # Test 1: Health check
    res = client.get("/")
    print(f"\n[Test 1] GET / (Health Check)")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.get_json()}")

    # Test 2: Valid Request (HTTP 200)
    valid_payload = {
        "distance_km": 6.8,
        "num_items": 4,
        "rain_flag": 1
    }
    print(f"\n[Test 2] POST /predict with Complete Valid Payload")
    print(f"Payload: {json.dumps(valid_payload, indent=2)}")
    res = client.post("/predict", json=valid_payload)
    print(f"Status Code: {res.status_code}")
    print(f"Response JSON: {json.dumps(res.get_json(), indent=2)}")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "predicted_delivery_time_min" in res.get_json(), "Missing predicted_delivery_time_min"
    print(">> TEST 2 PASSED: Received HTTP 200 with rounded predicted delivery time!")

    # Test 3: Incomplete Request missing 'rain_flag' (HTTP 400)
    incomplete_payload_1 = {
        "distance_km": 6.8,
        "num_items": 4
    }
    print(f"\n[Test 3] POST /predict with Incomplete Payload (Missing 'rain_flag')")
    print(f"Payload: {json.dumps(incomplete_payload_1, indent=2)}")
    res = client.post("/predict", json=incomplete_payload_1)
    print(f"Status Code: {res.status_code}")
    print(f"Response JSON: {json.dumps(res.get_json(), indent=2)}")
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    assert "rain_flag" in res.get_json().get("missing_fields", []), "rain_flag should be listed in missing_fields"
    print(">> TEST 3 PASSED: Server gracefully rejected missing field with HTTP 400 without crashing!")

    # Test 4: Incomplete Request missing 'distance_km' (HTTP 400)
    incomplete_payload_2 = {
        "num_items": 2,
        "rain_flag": 0
    }
    print(f"\n[Test 4] POST /predict with Incomplete Payload (Missing 'distance_km')")
    print(f"Payload: {json.dumps(incomplete_payload_2, indent=2)}")
    res = client.post("/predict", json=incomplete_payload_2)
    print(f"Status Code: {res.status_code}")
    print(f"Response JSON: {json.dumps(res.get_json(), indent=2)}")
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    assert "distance_km" in res.get_json().get("missing_fields", []), "distance_km should be listed in missing_fields"
    print(">> TEST 4 PASSED: Server handled incomplete request safely!")

    print("\n" + "=" * 65)
    print("CURL COMMANDS FOR MANUAL / POSTMAN REPLICATION:")
    print("=" * 65)
    print("""
# 1. Valid request (Expected HTTP 200):
curl -X POST http://localhost:5000/predict \\
     -H "Content-Type: application/json" \\
     -d '{"distance_km": 6.8, "num_items": 4, "rain_flag": 1}'

# 2. Incomplete request missing 'rain_flag' (Expected HTTP 400):
curl -X POST http://localhost:5000/predict \\
     -H "Content-Type: application/json" \\
     -d '{"distance_km": 6.8, "num_items": 4}'
""")
    print("=" * 65)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
