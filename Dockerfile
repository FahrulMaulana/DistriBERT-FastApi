FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download DistilBERT model to avoid download during runtime
RUN python -c "from transformers import DistilBertTokenizer, DistilBertForSequenceClassification; \
    print('Downloading DistilBERT model...'); \
    DistilBertTokenizer.from_pretrained('distilbert-base-uncased'); \
    DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased'); \
    print('Model download completed!')"

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p ./models/cache ./logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "main.py"]
