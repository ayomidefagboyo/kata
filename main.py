"""
Kata Application Entrypoint
Exposes the FastAPI application instance for Vercel, Uvicorn, and ASGI runtimes.
"""
from kata.main import app

__all__ = ["app"]
