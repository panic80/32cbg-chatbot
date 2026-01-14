# Dockerfile for RAG Service (Python FastAPI)
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/app/nltk_data

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --timeout=100 --retries=5 -r requirements.txt

# Download NLTK data to the shared location
RUN mkdir -p /app/nltk_data && \
    python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data'); nltk.download('punkt_tab', download_dir='/app/nltk_data'); nltk.download('stopwords', download_dir='/app/nltk_data'); nltk.download('averaged_perceptron_tagger', download_dir='/app/nltk_data'); nltk.download('averaged_perceptron_tagger_eng', download_dir='/app/nltk_data')"

# Copy application code
COPY app/ ./app/
COPY evaluation/ ./evaluation/

# Create directories for data, chroma_db, and logs, and set ownership
RUN mkdir -p /app/data /app/chroma_db /app/logs /var/log/cbthis /home/appuser && \
    chown -R appuser:appuser /app /var/log/cbthis /home/appuser

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
