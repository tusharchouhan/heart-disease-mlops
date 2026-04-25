"""
Data preprocessing and feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os


def load_and_clean_data(filepath="data/heart.csv"):
    """Load and clean the heart disease dataset."""
    df = pd.read_csv(filepath)
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df


def get_feature_lists():
    """Return lists of numerical and categorical features."""
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    return numerical_features, categorical_features


def build_preprocessor():
    """Build a sklearn preprocessing pipeline."""
    numerical_features, categorical_features = get_feature_lists()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_features)
        ]
    )
    
    return preprocessor


def prepare_data(filepath="data/heart.csv", test_size=0.2, random_state=42):
    """
    Full data preparation pipeline.
    Returns X_train, X_test, y_train, y_test, preprocessor
    """
    df = load_and_clean_data(filepath)
    
    numerical_features, categorical_features = get_feature_lists()
    all_features = numerical_features + categorical_features
    
    X = df[all_features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    preprocessor = build_preprocessor()
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def save_preprocessor(preprocessor, path="models/preprocessor.joblib"):
    """Save the preprocessing pipeline."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved to {path}")


def load_preprocessor(path="models/preprocessor.joblib"):
    """Load a saved preprocessing pipeline."""
    return joblib.load(path)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    save_preprocessor(preprocessor)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training target distribution:\n{y_train.value_counts()}")