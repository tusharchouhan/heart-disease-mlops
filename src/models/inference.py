"""
Inference module for heart disease prediction.
Provides a complete, reproducible prediction pipeline.
"""

import joblib
import numpy as np
import pandas as pd
import os


class HeartDiseasePredictor:
    """
    Complete prediction pipeline for heart disease classification.
    Handles preprocessing and prediction in a single interface.
    """
    
    def __init__(self, model_path="models/best_model.joblib", 
                 preprocessor_path="models/preprocessor.joblib"):
        """Load the trained model and preprocessor."""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_names = [
            'age', 'trestbps', 'chol', 'thalach', 'oldpeak',  # numerical
            'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'  # categorical
        ]
    
    def predict(self, input_data: dict) -> dict:
        """
        Make a prediction from a dictionary of features.
        
        Args:
            input_data: dict with feature names as keys
            
        Returns:
            dict with prediction and confidence
        """
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Ensure correct column order
        df = df[self.feature_names]
        
        # Preprocess
        X_processed = self.preprocessor.transform(df)
        
        # Predict
        prediction = int(self.model.predict(X_processed)[0])
        probability = float(self.model.predict_proba(X_processed)[0][1])
        
        result = {
            'prediction': prediction,
            'prediction_label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
            'confidence': round(probability, 4),
            'risk_level': self._get_risk_level(probability)
        }
        
        return result
    
    def _get_risk_level(self, probability: float) -> str:
        """Categorize risk level based on probability."""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.6:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def predict_batch(self, input_list: list) -> list:
        """Make predictions for multiple inputs."""
        return [self.predict(item) for item in input_list]


def get_sample_input():
    """Return a sample input for testing."""
    return {
        'age': 55,
        'sex': 1,
        'cp': 2,
        'trestbps': 130,
        'chol': 250,
        'fbs': 0,
        'restecg': 1,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 1.5,
        'slope': 1,
        'ca': 0,
        'thal': 3
    }


if __name__ == "__main__":
    print("=" * 50)
    print("HEART DISEASE PREDICTION - INFERENCE TEST")
    print("=" * 50)
    
    predictor = HeartDiseasePredictor()
    sample = get_sample_input()
    
    print(f"\nInput: {sample}")
    result = predictor.predict(sample)
    print(f"\nPrediction: {result['prediction_label']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Risk Level: {result['risk_level']}")