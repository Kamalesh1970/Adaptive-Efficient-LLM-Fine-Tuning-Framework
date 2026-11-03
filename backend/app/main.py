"""
FastAPI Application Entry Point for Adaptive LLM Fine-Tuning Studio.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.api.routes import router as api_router
from backend.app.services.job_manager import get_job_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager ensuring interrupted/orphaned runs are reconciled on startup."""
    try:
        mgr = get_job_manager()
        reconciled = mgr.reconcile_interrupted_runs()
        if reconciled > 0:
            logger.info(f"Startup reconciliation: reconciled {reconciled} orphaned training run(s).")
    except Exception as e:
        logger.warning(f"Error during startup run reconciliation: {e}")
    yield


def create_app() -> FastAPI:
    """Builds and configures the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Backend API for Adaptive LLM Fine-Tuning Studio (Phase 0 Foundation)",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS for Vite frontend development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    app.include_router(api_router)

    @app.get("/", tags=["Root"])
    def root_info():
        return {
            "app": settings.app_name,
            "version": settings.version,
            "status": "online",
            "docs": "/docs",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.host, port=settings.port, reload=True)
