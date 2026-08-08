"""
Application entrypoint.

Models (SCRFD, ArcFace, MiniFASNet) are loaded once at startup via the
lifespan context manager, not per-request -- this is the difference between
millisecond-level and multi-second authentication latency.
"""
from contextlib import asynccontextmanager
import logging
import time
import collections

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.ml.model_manager import ModelManager
from app.ml.pipeline import FacePipeline
from app.ml.runtime import get_inference_runtime
from app.logging_config import configure_logging
from app.routers import registration, authentication, health, models, monitoring
from app.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.LOG_LEVEL)
    started = time.perf_counter()
    id_migrations = await init_db()
    get_vector_store().migrate_student_ids(id_migrations)

    runtime = get_inference_runtime()
    model_manager = ModelManager()
    app.state.model_manager = model_manager
    app.state.pipeline = FacePipeline(model_manager)
    app.state.metrics = {"active_requests": 0, "latencies": collections.deque(maxlen=500), "faiss_latencies": collections.deque(maxlen=500)}
    logger.info(
        "application_ready",
        extra={
            "event": "application_ready", "device": runtime.device,
            "active_models": model_manager.status()["active"], "startup_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )

    yield

    app.state.pipeline = None
    app.state.model_manager = None


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)


@app.middleware("http")
async def log_api_request(request, call_next):
    started = time.perf_counter()
    is_authentication = request.url.path.endswith("/authenticate")
    if is_authentication and hasattr(request.app.state, "metrics"):
        request.app.state.metrics["active_requests"] += 1
    try:
        response = await call_next(request)
    except Exception:
        if is_authentication and hasattr(request.app.state, "metrics"):
            request.app.state.metrics["active_requests"] -= 1
            request.app.state.metrics["latencies"].append(round((time.perf_counter() - started) * 1000, 2))
        logger.exception(
            "api_request_failed",
            extra={"event": "api_request_failed", "method": request.method, "path": request.url.path},
        )
        raise
    latency = round((time.perf_counter() - started) * 1000, 2)
    if is_authentication and hasattr(request.app.state, "metrics"):
        request.app.state.metrics["active_requests"] -= 1
        request.app.state.metrics["latencies"].append(latency)
    logger.info(
        "api_request_completed",
        extra={
            "event": "api_request_completed", "method": request.method, "path": request.url.path,
            "status_code": response.status_code, "latency_ms": latency,
        },
    )
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(registration.router)
app.include_router(authentication.router)
app.include_router(models.router)
app.include_router(monitoring.router)
