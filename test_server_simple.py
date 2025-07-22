#!/usr/bin/env python3
"""Simple server test to verify FastAPI startup without lifespan issues."""

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add server path
sys.path.insert(0, str(Path(__file__).parent))

# Create simple FastAPI app without lifespan
app = FastAPI(
    title="GAJA Assistant Server - Simple Test",
    description="Simple test server without lifespan complications",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "GAJA Assistant Server - Simple Test", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "server": "simple-test"}


@app.get("/api/v1/health")
async def api_health():
    """API health check."""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    print("Starting simple test server...")
    uvicorn.run(
        app,
        host="localhost",
        port=8001,
        log_level="warning",  # Zmieniono z "info" na "warning" żeby ukryć zbędne logi
        access_log=False,     # Wyłączono logi żądań HTTP
        reload=False,
    )
    print("Server stopped")
