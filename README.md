# 🫀 Heart Disease Prediction — End-to-End MLOps Pipeline

[![CI/CD Pipeline](https://github.com/tusharchouhan/heart-disease-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/tusharchouhan/heart-disease-mlops/actions/workflows/ci-cd.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.19.0-blue.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, fully automated Machine Learning Operations (MLOps) pipeline for predicting heart disease risk using patient clinical data. Built with modern best practices including CI/CD, experiment tracking, containerization, cloud deployment, and monitoring.

---

##  Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Models and Results](#-models-and-results)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Experiment Tracking](#-experiment-tracking)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Docker](#-docker)
- [Cloud Deployment](#-cloud-deployment)
- [Monitoring](#-monitoring)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Author](#-author)

---

##  Overview

### Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early detection through risk assessment can significantly improve patient outcomes. This project builds a complete MLOps pipeline that:

- Analyzes patient health data (13 clinical features)
- Trains and compares 3 ML classification models
- Tracks all experiments with MLflow
- Runs 29 automated unit tests
- Automates the full pipeline with CI/CD (GitHub Actions)
- Containerizes the API with Docker
- Deploys to Google Cloud Run
- Monitors predictions with Prometheus metrics

### Key Highlights

| Feature | Details |
|---------|---------|
| Best Model | Logistic Regression (ROC-AUC: 0.962) |
| API Framework | FastAPI with interactive Swagger docs |
| Tests | 29 unit tests with 100% pass rate |
| CI/CD | 4-stage GitHub Actions pipeline |
| Deployment | Google Cloud Run with auto-scaling |
| Monitoring | Prometheus metrics and structured logging |

---

## 🏗 Architecture

```text
GitHub Repository
├── Source Code (src/)
│   ├── Data Pipeline ──→ Download, Clean, Preprocess
│   ├── Model Training ──→ 3 Models + MLflow Tracking
│   ├── API (FastAPI) ──→ /predict, /health, /metrics
│   └── Tests (Pytest) ──→ 29 Unit Tests
├── GitHub Actions CI/CD Pipeline
│   ├── Stage 1: Lint (Flake8)
│   ├── Stage 2: Test (Pytest - 29 tests)
│   ├── Stage 3: Train (MLflow tracking)
│   └── Stage 4: Docker Build and Test
├── Docker Container
├── Google Cloud Run
└── Monitoring
```

---

## 📂 Project Structure

```text
heart-disease-mlops/
├── .github/workflows/
├── data/
├── deployment/
├── models/
├── monitoring/
├── notebooks/
├── screenshots/
├── src/
├── tests/
├── Dockerfile
├── README.md
└── requirements.txt
```

---

##  Dataset

**Title:** Heart Disease UCI Dataset  
**Source:** https://archive.ics.uci.edu/ml/datasets/heart+Disease

- Records: 303 patients
- Features: 13 clinical attributes
- Target: Binary classification
- Missing Values: Median imputation

---

##  Models and Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 0.8689 | 0.9621 |
| Random Forest | 0.8361 | 0.9008 |
| Gradient Boosting | 0.8197 | 0.9020 |

### Best Model: Logistic Regression

Selected based on highest ROC-AUC and stable cross-validation.

---

## 🛠 Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11 |
| ML Framework | Scikit-learn | 1.5.2 |
| API | FastAPI | 0.115.6 |
| Server | Uvicorn | 0.34.0 |
| Experiment Tracking | MLflow | 2.19.0 |
| Data Processing | Pandas | 2.2.3 |
| Numerical Computing | NumPy | 2.1.3 |
| Visualization | Matplotlib, Seaborn | 3.9.3, 0.13.2 |
| Testing | Pytest | 8.3.4 |
| Linting | Flake8 | 7.1.1 |
| Containerization | Docker | Latest |
| CI/CD | GitHub Actions | v4 |
| Cloud | Google Cloud Run | Latest |
| Monitoring | Prometheus Client | 0.21.1 |

---

##  Quick Start

```bash
git clone https://github.com/tusharchouhan/heart-disease-mlops.git
cd heart-disease-mlops
python -m venv venv
pip install -r requirements.txt
python src/data/download_data.py
python src/models/train_mlflow.py
python -m pytest tests/ -v
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

---

##  API Documentation

### Endpoints

- `/` - API info
- `/health` - Health check
- `/predict` - Prediction endpoint
- `/metrics` - Prometheus metrics
- `/docs` - Swagger UI

---

##  Experiment Tracking

MLflow tracks:
- Parameters
- Metrics
- Artifacts
- Model versions

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

##  CI/CD Pipeline

Stages:
1. Lint
2. Test
3. Train
4. Docker Build & Test

---

##  Docker

```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8080 heart-disease-api:latest
```

---

## ☁ Cloud Deployment

Google Cloud Run with:
- Auto-scaling
- HTTPS endpoint
- GitHub integration

---

##  Monitoring

- Prometheus `/metrics`
- Structured logs
- Prediction latency
- Request counters

---

##  Testing

- 29 unit tests
- API tests
- Model tests
- Preprocessing tests

---

##  Troubleshooting

- Install dependencies: `pip install -e .`
- Download data before tests
- Check model files
- Resolve port conflicts

---

## 👤 Author

**Tushar Chouhan**  
Course: MLOps (S2-25_AMLCSZG523)  
GitHub: [![tusharchouhan](https://github.com/tusharchouhan)  
Repository: [![heart-disease-mlops](https://github.com/tusharchouhan/heart-disease-mlops)

---

##  Acknowledgments

- UCI Machine Learning Repository
- Scikit-learn
- FastAPI
- MLflow
- Docker
- Google Cloud
