# Dockerfile for Heart Disease Prediction API - GCP Cloud Run
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL application code
COPY . .

# Install package
RUN pip install -e .

# Download data and train model during build
RUN python src/data/download_data.py
RUN python src/models/train_mlflow.py

# Create log directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8080

# Run the application
CMD exec uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8080}
