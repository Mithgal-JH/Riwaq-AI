# Multi-stage production Dockerfile for BinX Post Recommendation Engine
# Stage 1: Build dependencies & pre-download neural model weights
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download SentenceTransformer model weights into local container cache
# Guarantees zero network calls and sub-second container initialization at runtime
ENV HF_HOME=/app/model_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Stage 2: Final hardened runtime image
FROM python:3.11-slim

WORKDIR /app

# Security: Create non-root system user and group (UID 10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -m appuser

# Install curl for Docker health check
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies and model weights from builder
COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /app/model_cache /app/model_cache

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/model_cache \
    TRANSFORMERS_OFFLINE=1 \
    PORT=8000

# Copy application code and mock catalog
COPY src/ /app/src/
COPY data/ /app/data/

# Ensure non-root ownership
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Healthcheck validating microservice availability
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
