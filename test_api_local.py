"""
Local API testing script.
Run this while the API is running on localhost:8000
"""

import requests
import json
from datetime import datetime


def test_api():
    BASE_URL = "http://localhost:8000"

    print("=" * 60)
    print("HEART DISEASE API - LOCAL TESTING")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    # Test 1: Root endpoint
    print("\n--- Test 1: Root Endpoint ---")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Health check
    print("\n--- Test 2: Health Check ---")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Prediction - High risk patient
    print("\n--- Test 3: Prediction (High Risk Patient) ---")
    high_risk = {
        "age": 65, "sex": 1, "cp": 3, "trestbps": 160,
        "chol": 300, "fbs": 1, "restecg": 2, "thalach": 120,
        "exang": 1, "oldpeak": 3.0, "slope": 0, "ca": 2, "thal": 7
    }
    try:
        response = requests.post(f"{BASE_URL}/predict", json=high_risk)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 4: Prediction - Low risk patient
    print("\n--- Test 4: Prediction (Low Risk Patient) ---")
    low_risk = {
        "age": 35, "sex": 0, "cp": 0, "trestbps": 110,
        "chol": 180, "fbs": 0, "restecg": 0, "thalach": 175,
        "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 3
    }
    try:
        response = requests.post(f"{BASE_URL}/predict", json=low_risk)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 5: Prediction - Medium risk patient
    print("\n--- Test 5: Prediction (Medium Risk Patient) ---")
    medium_risk = {
        "age": 55, "sex": 1, "cp": 2, "trestbps": 130,
        "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
        "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 3
    }
    try:
        response = requests.post(f"{BASE_URL}/predict", json=medium_risk)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 6: Invalid input (should return 422)
    print("\n--- Test 6: Invalid Input (Missing Fields) ---")
    invalid_input = {"age": 55, "sex": 1}
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_input)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 7: Metrics endpoint
    print("\n--- Test 7: Prometheus Metrics ---")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        print(f"Status: {response.status_code}")
        print("Metrics (first 500 chars):")
        print(response.text[:500])
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    test_api()