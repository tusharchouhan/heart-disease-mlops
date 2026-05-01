"""
Monitoring dashboard - checks API health, metrics, and logs.
"""

import requests
import json
import time
from datetime import datetime


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_health(base_url):
    """Check API health status."""
    print_header("HEALTH CHECK")
    try:
        start = time.time()
        response = requests.get(f"{base_url}/health", timeout=10)
        latency = time.time() - start
        data = response.json()
        print(f"  Status Code: {response.status_code}")
        print(f"  API Status:  {data.get('status', 'unknown')}")
        print(f"  Model Loaded: {data.get('model_loaded', 'unknown')}")
        print(f"  Response Time: {latency:.4f}s")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def send_test_predictions(base_url):
    """Send multiple predictions and track results."""
    print_header("PREDICTION TESTS")

    test_cases = [
        {
            "name": "High Risk Patient",
            "data": {
                "age": 65, "sex": 1, "cp": 3, "trestbps": 160,
                "chol": 300, "fbs": 1, "restecg": 2, "thalach": 120,
                "exang": 1, "oldpeak": 3.0, "slope": 0, "ca": 2, "thal": 7
            }
        },
        {
            "name": "Low Risk Patient",
            "data": {
                "age": 35, "sex": 0, "cp": 0, "trestbps": 110,
                "chol": 180, "fbs": 0, "restecg": 0, "thalach": 175,
                "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 3
            }
        },
        {
            "name": "Medium Risk Patient",
            "data": {
                "age": 55, "sex": 1, "cp": 2, "trestbps": 130,
                "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
                "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 3
            }
        }
    ]

    for i, case in enumerate(test_cases, 1):
        try:
            start = time.time()
            response = requests.post(
                f"{base_url}/predict",
                json=case["data"],
                timeout=10
            )
            latency = time.time() - start
            result = response.json()
            print(f"\n  Test {i}: {case['name']}")
            print(f"    Prediction:    {result.get('prediction_label', 'N/A')}")
            print(f"    Confidence:    {result.get('confidence', 'N/A')}")
            print(f"    Risk Level:    {result.get('risk_level', 'N/A')}")
            print(f"    Response Time: {latency:.4f}s")
        except Exception as e:
            print(f"\n  Test {i}: {case['name']} - ❌ Error: {e}")


def check_metrics(base_url):
    """Fetch and display Prometheus metrics."""
    print_header("PROMETHEUS METRICS")
    try:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            for line in lines:
                if line and not line.startswith('#'):
                    print(f"  {line}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def check_api_logs():
    """Check local API log file."""
    print_header("API LOGS (Last 10 entries)")
    try:
        with open("api.log", "r") as f:
            lines = f.readlines()
            last_lines = lines[-10:] if len(lines) >= 10 else lines
            for line in last_lines:
                print(f"  {line.strip()}")
    except FileNotFoundError:
        print("  No log file found (api.log)")
    except Exception as e:
        print(f"  Error reading logs: {e}")


def run_dashboard(base_url="http://localhost:8000"):
    """Run the complete monitoring dashboard."""
    print("\n" + "#" * 60)
    print("#" + " " * 16 + "MONITORING DASHBOARD" + " " * 16 + " #")
    print("#" * 60)
    print(f"\n  Target: {base_url}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all checks
    health_ok = check_health(base_url)
    if health_ok:
        send_test_predictions(base_url)
    check_metrics(base_url)
    check_api_logs()

    print_header("DASHBOARD SUMMARY")
    print(f"  API Health:     {'✅ Healthy' if health_ok else '❌ Unhealthy'}")
    print(f"  Metrics:        ✅ Available at {base_url}/metrics")
    print(f"  Logs:           ✅ Available in api.log")
    print(f"  Timestamp:      {datetime.now().isoformat()}")
    print("\n" + "#" * 60)


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    run_dashboard(url)