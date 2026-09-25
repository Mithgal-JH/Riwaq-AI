FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/ai-gateway/requirements.txt /build/ai-gateway-requirements.txt
COPY services/content-analysis/requirements.txt /build/content-analysis-requirements.txt
COPY services/post-recommendation/requirements.txt /build/post-recommendation-requirements.txt

RUN python -m venv /opt/venvs/ai-gateway && \
    /opt/venvs/ai-gateway/bin/pip install --no-cache-dir -r /build/ai-gateway-requirements.txt

RUN python -m venv /opt/venvs/content-analysis && \
    /opt/venvs/content-analysis/bin/pip install --no-cache-dir -r /build/content-analysis-requirements.txt

RUN python -m venv /opt/venvs/post-recommendation && \
    /opt/venvs/post-recommendation/bin/pip install --no-cache-dir -r /build/post-recommendation-requirements.txt

# Keep the existing post-recommendation startup behavior by baking in its model.
ENV HF_HOME=/opt/model_cache
RUN /opt/venvs/post-recommendation/bin/python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


FROM python:3.11-slim

WORKDIR /opt/riwaq

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid appgroup --shell /usr/sbin/nologin --create-home appuser

COPY --from=builder /opt/venvs /opt/venvs
COPY --from=builder /opt/model_cache /opt/model_cache
COPY services/ai-gateway/app /opt/riwaq/ai-gateway/app
COPY services/content-analysis/app /opt/riwaq/content-analysis/app
COPY services/post-recommendation/app /opt/riwaq/post-recommendation/app
COPY services/post-recommendation/data /opt/riwaq/post-recommendation/data
COPY scripts/start-riwaq.sh /usr/local/bin/start-riwaq.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/model_cache \
    TRANSFORMERS_OFFLINE=1

RUN chmod 755 /usr/local/bin/start-riwaq.sh && \
    chown -R appuser:appgroup /opt/riwaq /opt/model_cache /usr/local/bin/start-riwaq.sh

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f "http://127.0.0.1:${PORT:-8080}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/start-riwaq.sh"]