# Dockerfile for Ohrid Water Demand Research Framework
# Multi-stage build for production deployment

# Stage 1: Development and testing environment
FROM python:3.11-slim as development

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN curl https://sdk.cloud.google.com | bash
ENV PATH=$PATH:/root/google-cloud-sdk/bin

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/features data/external \
    models logs results

# Set environment variables
ENV PYTHONPATH=/app/src
ENV GOOGLE_CLOUD_PROJECT=""
ENV OPENWEATHER_API_KEY=""

# Development target
FROM development as dev
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 2: Production environment
FROM python:3.11-slim as production

# Set working directory
WORKDIR /app

# Install system dependencies (minimal for production)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK (minimal)
RUN curl https://sdk.cloud.google.com | bash
ENV PATH=$PATH:/root/google-cloud-sdk/bin

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary application files
COPY src/ ./src/
COPY config/ ./config/
COPY deploy_to_gcp.py .
COPY test_real_data_collection.py .

# Copy trained models and configurations
COPY models/ ./models/
COPY data/features/ ./data/features/

# Create runtime directories
RUN mkdir -p logs results data/raw data/processed data/external

# Set environment variables
ENV PYTHONPATH=/app/src
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import src.data_collectors.ohrid_data_manager; print('Health check passed')" || exit 1

# Production command
CMD ["python", "src/models/ohrid_predictor.py"]

# Stage 3: Training environment (for ML model training)
FROM development as training

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    optuna \
    mlflow \
    tensorboard

# Copy training data
COPY data/ ./data/

# Expose MLflow tracking port
EXPOSE 5000

# Training command
CMD ["python", "-m", "src.models.train_models"]

# Stage 4: API serving environment
FROM python:3.11-slim as api

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install FastAPI and related dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# Copy API code and models
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

# Set environment variables
ENV PYTHONPATH=/app/src

# Expose API port
EXPOSE 8000

# API command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]