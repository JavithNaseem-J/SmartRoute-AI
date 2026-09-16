import asyncio
import os
from pathlib import Path
from typing import Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from src.pipeline.inference import InferencePipeline
from src.utils.logger import logger


def _read_build_metadata_file(filename: str) -> str:
    try:
        return (Path("/app") / filename).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def deployment_commit_sha() -> str:
    metadata_sha = _read_build_metadata_file(".commit_sha")
    if metadata_sha and metadata_sha != "unknown":
        return metadata_sha

    for env_name in ("SMARTROUTE_COMMIT_SHA", "RENDER_GIT_COMMIT"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return metadata_sha or "unknown"


def deployment_build_time() -> str:
    value = os.getenv("SMARTROUTE_BUILD_TIME", "").strip()
    if value:
        return value
    return _read_build_metadata_file(".build_time") or "unknown"


def create_system_router(get_pipeline: Callable[[], InferencePipeline | None]) -> APIRouter:
    router = APIRouter(tags=["system"])

    async def component_status() -> dict:
        pipeline = get_pipeline()
        if not pipeline:
            return {"pipeline": "initializing"}

        components = {
            "router": "ok" if pipeline.router else "error",
            "model_manager": "ok" if pipeline.model_manager else "error",
            "retriever": "ok" if pipeline.retriever else "error",
            "cost_tracker": "ok" if pipeline.tracker else "error",
            "redis": "error",
            "qdrant": "error",
            "postgres": "error",
        }

        try:
            from src.core.dependencies import get_redis_client

            redis_client = get_redis_client()
            if await redis_client.ping():
                components["redis"] = "ok"
        except Exception as e:
            logger.warning(f"Redis readiness check failed: {e}")
            components["redis"] = "error"

        try:
            from src.core.dependencies import get_qdrant_client

            qdrant_client = get_qdrant_client()
            await qdrant_client.get_collections()
            components["qdrant"] = "ok"
        except Exception as e:
            logger.warning(f"Qdrant readiness check failed: {e}")
            components["qdrant"] = "error"

        try:
            if pipeline.tracker and pipeline.tracker.engine:

                def ping_db() -> None:
                    with pipeline.tracker.engine.connect() as conn:
                        conn.execute(text("SELECT 1"))

                await asyncio.to_thread(ping_db)
                components["postgres"] = "ok"
        except Exception as e:
            logger.warning(f"Postgres readiness check failed: {e}")
            components["postgres"] = "error"

        return components

    @router.get("/health")
    async def health_check():
        """Cheap liveness probe for Docker, Render, and load balancers."""
        return {
            "status": "healthy" if get_pipeline() else "starting",
            "version": "2.0.0",
        }

    @router.get("/version")
    async def version():
        """Unauthenticated deployment identity used by CI/CD verification."""
        return {
            "commit_sha": deployment_commit_sha(),
            "build_time": deployment_build_time(),
        }

    @router.get("/ready")
    async def readiness_check():
        """Readiness probe that checks runtime dependencies."""
        components = await component_status()
        ready = bool(get_pipeline()) and all(value == "ok" for value in components.values())
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "not_ready",
                "version": "2.0.0",
                "components": components,
            },
        )

    return router
