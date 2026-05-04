# 🫀 Heart Disease Prediction — End-to-End MLOps Pipeline

[![CI/CD Pipeline](https://github.com/tusharchouhan/heart-disease-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/tusharchouhan/heart-disease-mlops/actions/workflows/ci-cd.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.19.0-blue.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, fully automated Machine Learning Operations (MLOps) pipeline for predicting heart disease risk using patient clinical data. Built with modern best practices including CI/CD, experiment tracking, containerization, cloud deployment, and monitoring.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models & Results](#models--results)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Experiment Tracking](#experiment-tracking)
- [CI/CD Pipeline](#cicd-pipeline)
- [Docker](#docker)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Screenshots](#screenshots)
- [Author](#author)

---

## 🎯 Overview

### Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early detection through risk assessment can significantly improve patient outcomes. This project builds a **complete MLOps pipeline** that:

- 🔍 Analyzes patient health data (13 clinical features)
- 🤖 Trains and compares 3 ML classification models
- 📊 Tracks all experiments with MLflow
- 🧪 Runs 29 automated unit tests
- 🔄 Automates the full pipeline with CI/CD (GitHub Actions)
- 🐳 Containerizes the API with Docker
- ☁️ Deploys to Google Cloud Run
- 📈 Monitors predictions with Prometheus metrics

### Key Highlights

| Feature | Details |
|---------|---------|
| **Best Model** | Logistic Regression (ROC-AUC: 0.962) |
| **API Framework** | FastAPI with interactive Swagger docs |
| **Tests** | 29 unit tests (100% pass rate) |
| **CI/CD** | 4-stage GitHub Actions pipeline |
| **Deployment** | Google Cloud Run (auto-scaling) |
| **Monitoring** | Prometheus metrics + structured logging |

---

## 🏗️ Architecture
┌──────────────────────────────────────────────────────────────────┐
│ GitHub Repository │
│ │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ │ Data │ │ Model │ │ API │ │ Tests │ │
│ │ Pipeline │ │ Training │ │(FastAPI) │ │(Pytest) │ │
│ └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘ │
│ └──────────────┴─────────────┴─────────────┘ │
│ │ │
│ ┌──────────────────┴───────────────────┐ │
│ │ GitHub Actions CI/CD Pipeline │ │
│ │ [Lint] → [Test] → [Train] → [Docker] │ │
│ └──────────────────┬───────────────────┘ │
└───────────────────────────┼──────────────────────────────────────┘
│
┌───────▼────────┐
│ Docker Image │
│ (Cloud Build) │
└───────┬────────┘
│
┌───────▼────────┐
│ Google Cloud │
│ Run │
│ (Auto-scaling) │
└───────┬────────┘
│
┌───────▼────────┐
│ Public API │
│ ┌───────────┐ │
│ │ /predict │ │
│ │ /health │ │
│ │ /metrics │ │
│ │ /docs │ │
│ └───────────┘ │
└───────┬────────┘
│
┌───────▼────────┐
│ Monitoring │
│ Prometheus + │
│ Logging │
└────────────────┘

---

## 📂 Project Structure
heart-disease-mlops/
│
├── 📁 .github/workflows/
│ └── ci-cd.yml # GitHub Actions CI/CD pipeline (4 stages)
│
├── 📁 data/
│ └── heart.csv # Heart Disease UCI dataset
│
├── 📁 deployment/
│ ├── deployment.yaml # Kubernetes Deployment manifest
│ ├── service.yaml # Kubernetes Service (LoadBalancer)
│ └── ingress.yaml # Kubernetes Ingress configuration
│
├── 📁 models/
│ ├── best_model.joblib # Trained best model (Logistic Regression)
│ ├── preprocessor.joblib # Fitted preprocessing pipeline
│ └── experiment_results.csv # Model comparison results
│
├── 📁 monitoring/
│ ├── prometheus.yml # Prometheus scrape configuration
│ └── dashboard.py # Custom monitoring dashboard script
│
├── 📁 notebooks/
│ └── 01_eda.py # Exploratory Data Analysis script
│
├── 📁 screenshots/ # All project screenshots for reporting
│
├── 📁 src/
│ ├── 📁 api/
│ │ ├── init.py
│ │ └── app.py # FastAPI application with /predict endpoint
│ ├── 📁 data/
│ │ ├── init.py
│ │ ├── download_data.py # Automated data download script
│ │ └── preprocess.py # Data cleaning & feature engineering
│ └── 📁 models/
│ ├── init.py
│ ├── train.py # Model training & evaluation
│ ├── train_mlflow.py # MLflow experiment tracking
│ └── inference.py # Production inference pipeline
│
├── 📁 tests/
│ ├── init.py
│ ├── test_preprocess.py # Data preprocessing tests (9 tests)
│ ├── test_model.py # Model training tests (10 tests)
│ └── test_api.py # API endpoint tests (6 tests)
│
├── .dockerignore # Docker build exclusions
├── .gitignore # Git exclusions
├── conftest.py # Pytest configuration
├── Dockerfile # Docker container definition
├── Procfile # Cloud deployment start command
├── README.md # This file
├── render.yaml # Render.com deployment config
├── requirements.txt # Python dependencies (pinned versions)
├── setup.cfg # Flake8 linting configuration
├── setup.py # Package installation setup
├── test_api_local.py # Local API testing script
└── test_deployed_api.py # Cloud API testing script

---

## 📊 Dataset

**Title:** Heart Disease UCI Dataset
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

| Property | Value |
|----------|-------|
| **Records** | 303 patients |
| **Features** | 13 clinical attributes |
| **Target** | Binary (0 = No Disease, 1 = Disease) |
| **Class Balance** | 54.1% No Disease, 45.9% Disease |
| **Missing Values** | Handled via median imputation |

### Features

| # | Feature | Description | Type |
|---|---------|-------------|------|
| 1 | `age` | Age in years | Numerical |
| 2 | `sex` | Sex (0=Female, 1=Male) | Categorical |
| 3 | `cp` | Chest pain type (0-3) | Categorical |
| 4 | `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| 5 | `chol` | Serum cholesterol (mg/dl) | Numerical |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl | Categorical |
| 7 | `restecg` | Resting ECG results (0-2) | Categorical |
| 8 | `thalach` | Maximum heart rate achieved | Numerical |
| 9 | `exang` | Exercise induced angina | Categorical |
| 10 | `oldpeak` | ST depression by exercise | Numerical |
| 11 | `slope` | Slope of peak exercise ST segment | Categorical |
| 12 | `ca` | Number of major vessels (0-4) | Categorical |
| 13 | `thal` | Thalassemia type | Categorical |

---

## 🏆 Models & Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** ⭐ | **0.8689** | **0.8333** | **0.8929** | **0.8621** | **0.9621** |
| Random Forest | 0.8361 | 0.8276 | 0.8571 | 0.8421 | 0.9008 |
| Gradient Boosting | 0.8197 | 0.8148 | 0.8462 | 0.8302 | 0.9020 |

### Cross-Validation Results (5-Fold Stratified)

| Model | CV ROC-AUC Mean | CV ROC-AUC Std |
|-------|-----------------|----------------|
| **Logistic Regression** ⭐ | **0.9021** | **±0.0165** |
| Random Forest | 0.8856 | ±0.0312 |
| Gradient Boosting | 0.8901 | ±0.0289 |

### Why Logistic Regression Won

- Highest ROC-AUC (0.962) on the test set
- Most stable cross-validation performance (lowest std: ±0.016)
- Best generalization despite being the simplest model
- Features are linearly separable after proper preprocessing

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.11 |
| **ML Framework** | Scikit-learn | 1.5.2 |
| **API** | FastAPI | 0.115.6 |
| **API Server** | Uvicorn | 0.34.0 |
| **Experiment Tracking** | MLflow | 2.19.0 |
| **Data Processing** | Pandas, NumPy | 2.2.3, 2.1.3 |
| **Visualization** | Matplotlib, Seaborn | 3.9.3, 0.13.2 |
| **Testing** | Pytest | 8.3.4 |
| **Linting** | Flake8 | 7.1.1 |
| **Containerization** | Docker | Latest |
| **CI/CD** | GitHub Actions | v4 |
| **Cloud** | Google Cloud Run | Latest |
| **Monitoring** | Prometheus Client | 0.21.1 |
| **Serialization** | Joblib | 1.4.2 |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git
- (Optional) Google Cloud SDK for cloud deployment

### 1. Clone the Repository

git clone https://github.com/tusharchouhan/heart-disease-mlops.git
cd heart-disease-mlops

2. Create Virtual Environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
source venv/bin/activate

3. Install Dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

4. Download Dataset
python src/data/download_data.py

5. Run Exploratory Data Analysis
python notebooks/01_eda.py

6. Train Models with Experiment Tracking
python src/models/train_mlflow.py

7. View MLflow Dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000

8. Run Tests
python -m pytest tests/ -v

9. Start the API
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs

10. Run Monitoring Dashboard
# In a separate terminal (while API is running)
python monitoring/dashboard.py


📡 API Documentation

Endpoints

Endpoint	Method	Description	Auth
/	GET	API info and available endpoints	None
/health	GET	Health check with model status	None
/predict	POST	Heart disease prediction	None
/metrics	GET	Prometheus monitoring metrics	None
/docs	GET	Interactive Swagger UI	None

Prediction Request

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

Prediction Response

json
{
  "prediction": 1,
  "prediction_label": "Heart Disease",
  "confidence": 0.8547,
  "risk_level": "High Risk",
  "timestamp": "2026-05-04T10:30:00.000000"
}

Risk Levels

Confidence Score	Risk Level
< 0.30	🟢 Low Risk
0.30 - 0.60	🟡 Medium Risk
> 0.60	🔴 High Risk

Health Check Response

json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "timestamp": "2026-05-04T10:30:00.000000"
}


Experiment Tracking

MLflow tracks all training experiments with:


Logged Items

Category	Items
Parameters	model_type, hyperparameters, test_size, random_state
Metrics	accuracy, precision, recall, f1_score, roc_auc, cv_scores
Artifacts	trained model, confusion matrix, ROC curve, classification report

View Experiments

# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Open in browser
http://127.0.0.1:5000


CI/CD Pipeline

The GitHub Actions pipeline runs automatically on every push to main:


┌─────────┐     ┌─────────┐     ┌─────────┐     ┌──────────────┐
│  LINT   │ ──▶ │  TEST   │ ──▶ │  TRAIN  │ ──▶ │ DOCKER BUILD │
│ Flake8  │     │ 29 Tests│     │ MLflow  │     │   & TEST     │
└─────────┘     └─────────┘     └─────────┘     └──────────────┘

Stage Details

Stage	Tool	What It Does
Lint	Flake8	Checks code quality and style
Test	Pytest	Runs 29 unit tests with coverage
Train	MLflow	Downloads data, trains 3 models, logs experiments
Docker	Docker	Builds image, starts container, tests all API endpoints

Pipeline Features

✅ Runs on every push to main and develop branches
✅ Runs on pull requests to main
✅ Fails fast on code errors or test failures
✅ Uploads test coverage and model artifacts
✅ Retry logic for Docker container health checks
✅ Complete logs for debugging


🐳 Docker

Build Locally

docker build -t heart-disease-api:latest .

Run Locally

docker run -d -p 8000:8080 --name heart-api heart-disease-api:latest

Test

# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.5,"slope":1,"ca":0,"thal":3}'

Docker Image Details

Property	Value
Base Image	python:3.11-slim
Size	~500MB
Port	8080
Health Check	Built-in (every 30s)
Data	Downloaded during build
Model	Trained during build


☁️ Cloud Deployment

Google Cloud Run

The API is deployed to Google Cloud Run with continuous deployment:


Property	Value
Platform	Google Cloud Run
Region	us-central1
Memory	512 MiB
CPU	1 vCPU
Auto-scaling	0 to 2 instances
Continuous Deploy	Yes (on git push)

Kubernetes Manifests

Production-ready K8s manifests are provided in deployment/:


# Deploy to any Kubernetes cluster
kubectl apply -f deployment/deployment.yaml    # 2 replicas with health probes
kubectl apply -f deployment/service.yaml       # LoadBalancer on port 80
kubectl apply -f deployment/ingress.yaml       # Nginx ingress


📈 Monitoring

Prometheus Metrics

Available at the /metrics endpoint:


Metric	Type	Description
predictions_total	Counter	Total predictions by result label
prediction_latency_seconds	Histogram	Processing time per prediction
http_requests_total	Counter	All HTTP requests by method/endpoint/status

Application Logging

Console: Real-time logs to stdout
File: Persistent logs to api.log
Format: timestamp - module - level - message
Details: Every prediction logs input, output, confidence, and latency

Monitoring Dashboard

# Run the monitoring dashboard (while API is running)
python monitoring/dashboard.py


🧪 Testing

Test Suite Summary

File	Tests	Coverage
test_preprocess.py	9 tests	Data loading, cleaning, feature engineering
test_model.py	10 tests	Training, evaluation, saving/loading
test_api.py	6 tests	All API endpoints, input validation
Total	29 tests	100% pass rate

Run Tests

# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_api.py -v

Test Categories

Data Tests:


Data loads correctly as DataFrame
No missing values after cleaning
Target is binary (0/1)
All expected columns present

Model Tests:


At least 2 models available
Each model trains successfully
Accuracy above 60% threshold
Predictions are binary
Probabilities in [0, 1] range
Models save and load correctly

API Tests:


Health endpoint returns 200
Predict returns valid predictions
Invalid input returns 422
Confidence scores in valid range


📸 Screenshots

All screenshots are available in the screenshots/ directory:


Screenshot	Description
01_class_distribution.png	Target variable distribution
02_feature_histograms.png	All feature distributions
03_correlation_heatmap.png	Feature correlation matrix
04_age_distribution.png	Age distribution by disease status
05_boxplots.png	Key features by target
06_model_comparison.png	Model performance comparison
cm_*.png	Confusion matrices for each model
mlflow_*.png	MLflow dashboard and run details
api_*.png	API documentation and responses
github_actions_*.png	CI/CD pipeline screenshots
monitoring_*.png	Monitoring dashboard output


🔧 Troubleshooting

Issue	Solution
ModuleNotFoundError	Run pip install -e . and activate venv
Tests fail	Run python src/data/download_data.py first
MLflow UI empty	Use mlflow ui --backend-store-uri sqlite:///mlflow.db
Port already in use	Kill process or change port number
Docker build fails	Ensure model files exist, check .dockerignore
API returns 500	Check that models/ has .joblib files


📄 License

This project is developed as part of the MLOps (S2-25_AMLCSZG523) course assignment.



👤 Author

Tushar Chouhan


📧 Course: MLOps (S2-25_AMLCSZG523)
🔗 GitHub: tusharchouhan
📦 Repository: heart-disease-mlops