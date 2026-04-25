"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.preprocess import (
    load_and_clean_data, get_feature_lists, 
    build_preprocessor, prepare_data
)


class TestDataLoading:
    """Tests for data loading and cleaning."""
    
    def test_load_data_returns_dataframe(self):
        """Test that data loads as a pandas DataFrame."""
        df = load_and_clean_data("data/heart.csv")
        assert isinstance(df, pd.DataFrame)
    
    def test_load_data_not_empty(self):
        """Test that loaded data is not empty."""
        df = load_and_clean_data("data/heart.csv")
        assert len(df) > 0
    
    def test_load_data_has_target_column(self):
        """Test that the target column exists."""
        df = load_and_clean_data("data/heart.csv")
        assert 'target' in df.columns
    
    def test_no_missing_values_after_cleaning(self):
        """Test that there are no missing values after cleaning."""
        df = load_and_clean_data("data/heart.csv")
        assert df.isnull().sum().sum() == 0
    
    def test_target_is_binary(self):
        """Test that target column contains only 0 and 1."""
        df = load_and_clean_data("data/heart.csv")
        unique_values = set(df['target'].unique())
        assert unique_values.issubset({0, 1})
    
    def test_data_has_expected_columns(self):
        """Test that all expected columns are present."""
        df = load_and_clean_data("data/heart.csv")
        expected_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                        'ca', 'thal', 'target']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"


class TestFeatureEngineering:
    """Tests for feature engineering."""
    
    def test_feature_lists_not_empty(self):
        """Test that feature lists are not empty."""
        num_features, cat_features = get_feature_lists()
        assert len(num_features) > 0
        assert len(cat_features) > 0
    
    def test_no_overlap_in_feature_lists(self):
        """Test that numerical and categorical features don't overlap."""
        num_features, cat_features = get_feature_lists()
        overlap = set(num_features) & set(cat_features)
        assert len(overlap) == 0, f"Overlapping features: {overlap}"
    
    def test_preprocessor_builds(self):
        """Test that preprocessor can be built."""
        preprocessor = build_preprocessor()
        assert preprocessor is not None


class TestDataPreparation:
    """Tests for the full data preparation pipeline."""
    
    def test_prepare_data_returns_correct_types(self):
        """Test that prepare_data returns correct types."""
        X_train, X_test, y_train, y_test, preprocessor = prepare_data()
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
    
    def test_train_test_split_sizes(self):
        """Test that train/test split is approximately 80/20."""
        X_train, X_test, y_train, y_test, preprocessor = prepare_data()
        total = len(y_train) + len(y_test)
        test_ratio = len(y_test) / total
        assert 0.15 <= test_ratio <= 0.25
    
    def test_no_data_leakage(self):
        """Test that train and test sets have correct sizes."""
        X_train, X_test, y_train, y_test, preprocessor = prepare_data()
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    def test_preprocessed_data_no_nan(self):
        """Test that preprocessed data has no NaN values."""
        X_train, X_test, y_train, y_test, preprocessor = prepare_data()
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()