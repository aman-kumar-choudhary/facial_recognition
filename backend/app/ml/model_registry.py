"""One shared InsightFace model pack for all face operations.

Loading ``FaceAnalysis`` is also InsightFace's model-download mechanism.  By
owning it in this process-wide registry, Antelope is downloaded and initialized
once, while detector and recognizer wrappers use its already-prepared objects.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from insightface.app import FaceAnalysis

from app.config import settings
from app.ml.runtime import InferenceRuntime, assert_cuda_session, get_inference_runtime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceModelRegistry:
    """Prepared detection and recognition models from one Antelope pack."""

    analysis: FaceAnalysis
    detector: object
    recognizer: object
    runtime: InferenceRuntime
    pack_name: str

    @classmethod
    def load(cls) -> "FaceModelRegistry":
        runtime = get_inference_runtime()
        pack_name = settings.FACE_MODEL_PACK or settings.RECOGNITION_MODEL_PACK
        started = time.perf_counter()
        logger.info(
            "model_pack_loading",
            extra={"event": "model_pack_loading", "pack": pack_name, "device": runtime.device},
        )
        # This is the only call that can trigger an InsightFace pack download.
        analysis = FaceAnalysis(
            name=pack_name,
            root=settings.INSIGHTFACE_MODEL_ROOT,
            allowed_modules=["detection", "recognition"],
            providers=runtime.providers,
        )
        analysis.prepare(
            ctx_id=runtime.ctx_id,
            det_thresh=settings.DETECTION_MIN_CONFIDENCE,
            det_size=(640, 640),
        )
        detector = analysis.models.get("detection")
        recognizer = analysis.models.get("recognition")
        if detector is None or recognizer is None:
            raise RuntimeError(
                f"Model pack '{pack_name}' must provide both detection and recognition models."
            )
        for model_name, model in (("SCRFD detector", detector), ("ArcFace recognizer", recognizer)):
            session = getattr(model, "session", None)
            if session is None:
                raise RuntimeError(f"{model_name} does not expose an ONNX Runtime session.")
            assert_cuda_session(session, model_name)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "model_pack_initialized",
            extra={
                "event": "model_pack_initialized",
                "pack": pack_name,
                "device": runtime.device,
                "latency_ms": elapsed_ms,
            },
        )
        return cls(analysis, detector, recognizer, runtime, pack_name)


_registry: FaceModelRegistry | None = None
_registry_lock = threading.Lock()


def get_face_model_registry() -> FaceModelRegistry:
    """Return the single prepared model registry for this API process."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = FaceModelRegistry.load()
    return _registry
