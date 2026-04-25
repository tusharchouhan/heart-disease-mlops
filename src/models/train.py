"""
Model training, evaluation, and comparison.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.preprocess import prepare_data, save_preprocessor


def get_models():
    """Return a dictionary of models to train."""
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42, learning_rate=0.1
        )
    }
    return models


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    return metrics, y_pred, y_prob


def cross_validate_model(model, X_train, y_train, cv=5):
    """Perform cross-validation and return scores."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = {}
    
    for metric in scoring_metrics:
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=metric)
        cv_results[f'cv_{metric}_mean'] = scores.mean()
        cv_results[f'cv_{metric}_std'] = scores.std()
    
    return cv_results


def plot_confusion_matrix(y_test, y_pred, model_name, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_model_comparison(results_df, save_path):
    """Plot comparison of all models."""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics_to_plot]
        ax.bar(x + i * width, values, width, label=row['model'], alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_and_evaluate_all():
    """Train all models and return comparison results."""
    print("=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    
    # Prepare data
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    save_preprocessor(preprocessor)
    
    models = get_models()
    all_results = []
    best_score = 0
    best_model_name = ""
    best_model = None
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        
        # Cross-validation
        cv_results = cross_validate_model(model, X_train, y_train)
        
        # Print results
        print(f"  Test Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Test Precision: {metrics['precision']:.4f}")
        print(f"  Test Recall:    {metrics['recall']:.4f}")
        print(f"  Test F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Test ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  CV Accuracy:    {cv_results['cv_accuracy_mean']:.4f} ± {cv_results['cv_accuracy_std']:.4f}")
        
        # Classification Report
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))
        
        # Save confusion matrix
        os.makedirs("screenshots", exist_ok=True)
        plot_confusion_matrix(y_test, y_pred, name, f"screenshots/cm_{name}.png")
        
        # Collect results
        result = {'model': name}
        result.update(metrics)
        result.update(cv_results)
        all_results.append(result)
        
        # Track best model
        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_model_name = name
            best_model = model
    
    # Results comparison
    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string(index=False))
    
    # Plot comparison
    plot_model_comparison(results_df, "screenshots/06_model_comparison.png")
    
    # Save best model
    print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {best_score:.4f})")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.joblib")
    print("Best model saved to models/best_model.joblib")
    
    return best_model, best_model_name, results_df, preprocessor


if __name__ == "__main__":
    best_model, best_model_name, results_df, preprocessor = train_and_evaluate_all()