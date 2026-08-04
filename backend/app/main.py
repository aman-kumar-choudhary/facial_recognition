"""
Application entrypoint.

Models (SCRFD, ArcFace, MiniFASNet) are loaded once at startup via the
lifespan context manager, not per-request -- this is the difference between
millisecond-level and multi-second authentication latency.
"""
from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.ml.face_detector import FaceDetector
from app.ml.face_recognizer import FaceRecognizer
from app.ml.liveness import LivenessDetector
from app.ml.model_registry import get_face_model_registry
from app.ml.pipeline import FacePipeline
from app.ml.runtime import get_inference_runtime
from app.logging_config import configure_logging
from app.routers import registration, authentication, health
from app.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.LOG_LEVEL)
    started = time.perf_counter()
    id_migrations = await init_db()
    get_vector_store().migrate_student_ids(id_migrations)

    runtime = get_inference_runtime()
    registry = get_face_model_registry()
    detector = FaceDetector(registry)
    recognizer = FaceRecognizer(registry)
    liveness = LivenessDetector()
    app.state.pipeline = FacePipeline(detector, recognizer, liveness)
    logger.info(
        "application_ready",
        extra={
            "event": "application_ready", "device": runtime.device, "face_model_pack": registry.pack_name,
            "liveness_models": liveness.model_count, "startup_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )

    yield

    app.state.pipeline = None


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)


@app.middleware("http")
async def log_api_request(request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "api_request_failed",
            extra={"event": "api_request_failed", "method": request.method, "path": request.url.path},
        )
        raise
    logger.info(
        "api_request_completed",
        extra={
            "event": "api_request_completed", "method": request.method, "path": request.url.path,
            "status_code": response.status_code, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
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
