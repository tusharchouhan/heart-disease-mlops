"""
Unit tests for model training and inference.
"""

import pytest
import numpy as np
import os
import sys
import joblib

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.preprocess import prepare_data
from src.models.train import get_models, evaluate_model, cross_validate_model


class TestModelTraining:
    """Tests for model training."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        self.X_train, self.X_test, self.y_train, self.y_test, self.preprocessor = prepare_data()
    
    def test_models_dict_not_empty(self):
        """Test that models dictionary is not empty."""
        models = get_models()
        assert len(models) >= 2, "Need at least 2 models"
    
    def test_logistic_regression_trains(self):
        """Test that Logistic Regression trains successfully."""
        models = get_models()
        model = models['LogisticRegression']
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        assert len(predictions) == len(self.y_test)
    
    def test_random_forest_trains(self):
        """Test that Random Forest trains successfully."""
        models = get_models()
        model = models['RandomForest']
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        assert len(predictions) == len(self.y_test)
    
    def test_model_accuracy_above_threshold(self):
        """Test that models achieve minimum accuracy."""
        models = get_models()
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            metrics, _, _ = evaluate_model(model, self.X_test, self.y_test)
            assert metrics['accuracy'] > 0.6, f"{name} accuracy too low: {metrics['accuracy']}"
    
    def test_model_predictions_are_binary(self):
        """Test that predictions are binary (0 or 1)."""
        models = get_models()
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)
            unique_preds = set(predictions)
            assert unique_preds.issubset({0, 1}), f"{name} non-binary predictions"
    
    def test_model_probabilities_in_range(self):
        """Test that predicted probabilities are between 0 and 1."""
        models = get_models()
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            probs = model.predict_proba(self.X_test)[:, 1]
            assert np.all(probs >= 0) and np.all(probs <= 1), f"{name} probabilities out of range"
    
    def test_evaluate_model_returns_expected_metrics(self):
        """Test that evaluation returns all expected metrics."""
        models = get_models()
        model = list(models.values())[0]
        model.fit(self.X_train, self.y_train)
        metrics, _, _ = evaluate_model(model, self.X_test, self.y_test)
        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
    
    def test_cross_validation_returns_results(self):
        """Test that cross-validation returns results."""
        models = get_models()
        model = list(models.values())[0]
        cv_results = cross_validate_model(model, self.X_train, self.y_train, cv=3)
        assert 'cv_accuracy_mean' in cv_results
        assert cv_results['cv_accuracy_mean'] > 0


class TestModelSaving:
    """Tests for model saving and loading."""
    
    def test_saved_model_exists(self):
        """Test that model file can be saved."""
        X_train, X_test, y_train, y_test, preprocessor = prepare_data()
        models = get_models()
        model = list(models.values())[0]
        model.fit(X_train, y_train)
        
        test_path = "models/test_model.joblib"
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, test_path)
        assert os.path.exists(test_path)
        
        # Cleanup
        os.remove(test_path)
    
    def test_saved_model_loads_and_predicts(self):
        """Test that a saved model can be loaded and make predictions."""
        X_train, X_test, y_train, y_test, preprocessor = prepare_data()
        models = get_models()
        model = list(models.values())[0]
        model.fit(X_train, y_train)
        
        test_path = "models/test_model_load.joblib"
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, test_path)
        
        loaded_model = joblib.load(test_path)
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == len(y_test)
        
        # Cleanup
        os.remove(test_path)