# GAJA Server - Docker Configuration
FROM python:3.12-slim

# Metadata
LABEL name="gaja-server" \
      version="1.0.0-beta" \
      description="GAJA Assistant AI Server" \
      maintainer="GAJA Team"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Copy requirements first for better caching
COPY requirements_server.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_server.txt

# Debug: List installed packages and verify critical packages
RUN pip list | grep -E "(aiohttp|fastapi|uvicorn|scrapling)" || true
RUN python - << 'PY'
import sys
print('Python:', sys.version)
try:
    from scrapling.parser import Selector
    html = '<html><body><main><h1>T</h1><p>ok</p></main></body></html>'
    s = Selector(html)
    assert s.css_first('h1::text')
    print('Scrapling parser OK')
except Exception as e:
    print('Scrapling parser check failed:', e)
PY

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data cache

## Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    GAJA_ENV="docker"
# WEATHERAPI_KEY will be supplied at runtime via -e WEATHERAPI_KEY=... or docker secrets

# Expose port (application listens on 8001 internally to align with compose mapping)
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash gaja
RUN chown -R gaja:gaja /app
USER gaja

# Default command
CMD ["python", "start.py"]
