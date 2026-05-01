"""
Model training with MLflow experiment tracking.
"""

import mlflow
import mlflow.sklearn
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.preprocess import prepare_data, save_preprocessor


def train_with_mlflow():
    """Train all models with MLflow tracking."""

    # Create ALL required directories FIRST
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Use SQLite backend for reliability
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Set MLflow experiment
    mlflow.set_experiment("heart-disease-classification")

    # Prepare data
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    save_preprocessor(preprocessor)

    # Define models with their hyperparameters
    model_configs = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            'params': {'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'}
        },
        'RandomForest': {
            'model': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'params': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2}
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42, learning_rate=0.1
            ),
            'params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
        }
    }

    best_score = 0
    best_model_name = ""
    best_run_id = ""
    best_model = None
    all_results = []

    for name, config in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        with mlflow.start_run(run_name=name):
            model = config['model']

            # Log parameters
            mlflow.log_param("model_type", name)
            for param_name, param_value in config['params'].items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }

            # Cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
            metrics['cv_roc_auc_mean'] = cv_scores.mean()
            metrics['cv_roc_auc_std'] = cv_scores.std()

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                print(f"  {metric_name}: {metric_value:.4f}")

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Create and log confusion matrix plot
            try:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['No Disease', 'Disease'],
                            yticklabels=['No Disease', 'Disease'])
                ax.set_title(f'Confusion Matrix - {name}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                plt.tight_layout()

                cm_path = os.path.join("screenshots", f"mlflow_cm_{name}.png")
                plt.savefig(cm_path, dpi=150)
                plt.close()
                mlflow.log_artifact(cm_path)
                print(f"  Saved: {cm_path}")
            except Exception as e:
                print(f"  Warning: Could not save confusion matrix: {e}")
                plt.close('all')

            # Create and log ROC curve
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name=name)
                ax.set_title(f'ROC Curve - {name}')
                plt.tight_layout()

                roc_path = os.path.join("screenshots", f"mlflow_roc_{name}.png")
                plt.savefig(roc_path, dpi=150)
                plt.close()
                mlflow.log_artifact(roc_path)
                print(f"  Saved: {roc_path}")
            except Exception as e:
                print(f"  Warning: Could not save ROC curve: {e}")
                plt.close('all')

            # Log classification report
            try:
                report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])
                report_path = os.path.join("screenshots", f"report_{name}.txt")
                with open(report_path, 'w') as f:
                    f.write(report)
                mlflow.log_artifact(report_path)
            except Exception as e:
                print(f"  Warning: Could not save report: {e}")

            # Collect results
            result = {'model': name, 'run_id': mlflow.active_run().info.run_id}
            result.update(metrics)
            all_results.append(result)

            # Track best model
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model_name = name
                best_run_id = mlflow.active_run().info.run_id
                best_model = model

    # Print summary
    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("EXPERIMENT TRACKING SUMMARY")
    print("=" * 70)
    print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string(index=False))
    print(f"\nBest Model: {best_model_name} (ROC-AUC: {best_score:.4f})")
    print(f"Best Run ID: {best_run_id}")

    # Save best model
    joblib.dump(best_model, "models/best_model.joblib")
    print("Best model saved to models/best_model.joblib")

    # Save results
    results_df.to_csv("models/experiment_results.csv", index=False)

    return best_model, results_df


if __name__ == "__main__":
    best_model, results_df = train_with_mlflow()
    print("\nTo view MLflow UI, run:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("Then open http://127.0.0.1:5000 in your browser")