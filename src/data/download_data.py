"""
Script to download the Heart Disease UCI dataset.
"""
import os
import pandas as pd

def download_heart_disease_data(output_path="data/heart.csv"):
    """
    Download the Heart Disease UCI dataset from the UCI ML Repository.
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # URL for the heart disease dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Column names for the dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    print("Downloading Heart Disease UCI dataset...")
    df = pd.read_csv(url, names=column_names, na_values='?')
    
    # Convert target to binary (0 = no disease, 1 = disease)
    df['target'] = (df['target'] > 0).astype(int)
    
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


if __name__ == "__main__":
    download_heart_disease_data()