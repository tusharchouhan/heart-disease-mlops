"""
Unit tests for the FastAPI application.
"""

import pytest
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.preprocess import prepare_data, save_preprocessor
from src.models.train import get_models, evaluate_model
import joblib


def ensure_models_exist():
    """Ensure model and preprocessor files exist for testing."""
    model_path = "models/best_model.joblib"
    preprocessor_path = "models/preprocessor.joblib"

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        X_train, X_test, y_train, y_test, preprocessor = prepare_data()
        save_preprocessor(preprocessor)
        models = get_models()
        model = models['RandomForest']
        model.fit(X_train, y_train)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)


# Ensure models exist before importing API
ensure_models_exist()
from src.api.app import app

from fastapi.testclient import TestClient
client = TestClient(app)


class TestAPIEndpoints:
    """Tests for API endpoints."""

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_predict_endpoint_valid_input(self):
        """Test prediction with valid input."""
        sample_input = {
            "age": 55,
            "sex": 1,
            "cp": 2,
            "trestbps": 130,
            "chol": 250,
            "fbs": 0,
            "restecg": 1,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.5,
            "slope": 1,
            "ca": 0,
            "thal": 3
        }
        response = client.post("/predict", json=sample_input)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "prediction_label" in data
        assert "confidence" in data
        assert data["prediction"] in [0, 1]

    def test_predict_endpoint_returns_confidence(self):
        """Test that confidence score is between 0 and 1."""
        sample_input = {
            "age": 45,
            "sex": 0,
            "cp": 1,
            "trestbps": 120,
            "chol": 200,
            "fbs": 0,
            "restecg": 0,
            "thalach": 160,
            "exang": 0,
            "oldpeak": 0.5,
            "slope": 2,
            "ca": 0,
            "thal": 3
        }
        response = client.post("/predict", json=sample_input)
        data = response.json()
        assert 0 <= data["confidence"] <= 1

    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200


class TestAPIInputValidation:
    """Tests for API input validation."""

    def test_missing_fields(self):
        """Test that missing fields return an error."""
        incomplete_input = {"age": 55, "sex": 1}
        response = client.post("/predict", json=incomplete_input)
        assert response.status_code == 422

    def test_empty_request(self):
        """Test that empty request returns an error."""
        response = client.post("/predict", json={})
        assert response.status_code == 422