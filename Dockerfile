# ==============================================================================
# Adaptive LLM Fine-Tuning Studio - Backend Container
# ==============================================================================

# Build stage: Install wheel dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final runtime stage
FROM python:3.12-slim AS runner

# Create non-root system user and group
RUN groupadd -g 10001 studio && \
    useradd -u 10001 -g studio -s /bin/bash -m appuser

WORKDIR /app

# Install runtime dependencies (e.g., curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY backend /app/backend
COPY core /app/core
COPY pyproject.toml /app/pyproject.toml

# Prepare writable directories for runtime persistence
RUN mkdir -p /app/artifacts/runs /app/artifacts/datasets /app/data && \
    chown -R appuser:studio /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

# Container healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000

# Graceful signal-handling entrypoint
ENTRYPOINT ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
