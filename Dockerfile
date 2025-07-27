# GAJA Server - Docker Configuration
FROM python:3.11-slim

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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_server.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_server.txt

# Debug: List installed packages and verify critical packages
RUN pip list | grep -E "(aiohttp|fastapi|uvicorn)"
RUN python -c "import aiohttp, fastapi, uvicorn; print('Core packages verified')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data cache

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
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
