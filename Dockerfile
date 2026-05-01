FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV GIT_PYTHON_REFRESH=quiet
ENV MPLBACKEND=Agg

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY conftest.py .
COPY setup.py .
RUN pip install -e .

RUN mkdir -p screenshots models data logs

RUN python src/data/download_data.py
RUN python src/models/train_mlflow.py

EXPOSE 8080

CMD exec uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8080}