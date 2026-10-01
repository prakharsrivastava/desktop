# =============================================================================
# Dockerfile for Cortex Analyst API — Copay
# =============================================================================
# Build:  docker build -t cortex-analyst-copay .
# Run:    docker run -p 8080:8080 --env-file .env cortex-analyst-copay
# =============================================================================

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose port (AWS ECS/App Runner expects 8080)
EXPOSE 8080

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]