"""Runtime-selectable, cached inference models.

The manager deliberately owns construction of every model.  A request takes a
single immutable snapshot, which means a selection change affects the next
frame without allowing one frame to combine models from two configurations.
Optional ONNX model paths use the same small adapter contract as the bundled
InsightFace and MiniFASNet models.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from app.config import settings
from app.ml.face_detector import FaceDetector, MTCNNFaceDetector, RetinaFaceDetector
from app.ml.face_recognizer import DeepFaceRecognizer, FaceNetRecognizer, FaceRecognizer
from app.ml.liveness import LivenessDetector
from app.ml.model_metrics import ModelMetrics
from app.ml.model_registry import get_face_model_registry

DETECTION_MODELS = ("SCRFD", "RetinaFace", "MTCNN")
LIVENESS_MODELS = ("MiniFASNet", "CDCN")
RECOGNITION_MODELS = ("ArcFace", "DeepFace", "FaceNet", "VGGFace2")


@dataclass(frozen=True)
class ActiveModels:
    detection: str
    liveness: str
    recognition: str
    detector: FaceDetector
    liveness_detector: LivenessDetector
    recognizer: FaceRecognizer


class ModelUnavailableError(RuntimeError):
    pass


class ModelManager:
    """Caches model objects and changes the active names atomically.

    The alternate model names are registered now, but are intentionally only
    selectable when their deployment artifact is configured.  This avoids
    silently labelling SCRFD/ArcFace output as a different benchmark model.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.metrics = ModelMetrics()
        # facenet-pytorch and Keras honour these locations while downloading.
        # Set them before importing an optional backend so no weights leak to
        # a developer's home-directory cache or a container layer.
        os.environ.setdefault("TORCH_HOME", str(settings.model_root_path / "torch"))
        os.environ.setdefault("KERAS_HOME", str(settings.model_root_path / "keras"))
        os.environ.setdefault("DEEPFACE_HOME", str(settings.model_root_path / "deepface"))
        # InsightFace loads SCRFD and ArcFace together, so report it as one
        # shared allocation rather than falsely double-counting GPU memory.
        with self.metrics.measure_load("SCRFD + ArcFace (InsightFace pack)", stage="shared", shared=True):
            registry = get_face_model_registry()
        self.metrics.set_device(registry.runtime.device)
        # The three established implementations remain the backwards-
        # compatible defaults and are constructed exactly once at startup.
        self._cache = {
            ("detection", "SCRFD"): FaceDetector(registry),
            ("recognition", "ArcFace"): FaceRecognizer(registry),
        }
        with self.metrics.measure_load("MiniFASNet", stage="liveness"):
            self._cache[("liveness", "MiniFASNet")] = LivenessDetector()
        self._active = {"detection": "SCRFD", "liveness": "MiniFASNet", "recognition": "ArcFace"}

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "active": dict(self._active),
                "available": {
                    "detection": self._available("detection", DETECTION_MODELS),
                    "liveness": self._available("liveness", LIVENESS_MODELS),
                    "recognition": self._available("recognition", RECOGNITION_MODELS),
                },
            }

    def performance_status(self) -> list[dict[str, object]]:
        with self._lock:
            return self.metrics.status(dict(self._active), self._telemetry_information())

    def performance_summary(self) -> dict[str, float | int | None]:
        return self.metrics.summary()

    def _telemetry_information(self) -> dict[str, dict[str, object]]:
        """Expose only facts available from currently loaded model objects."""
        details: dict[str, dict[str, object]] = {
            "SCRFD + ArcFace (InsightFace pack)": {
                "model_type": "InsightFace Antelope model pack",
                "purpose": "Shared face detection and embedding models",
                "framework": "InsightFace / ONNX Runtime",
                "device": self.metrics.device,
                "models_included": "SCRFD, ArcFace",
            }
        }
        definitions = {
            "SCRFD": ("detection", "SCRFD", "SCRFD face detector", "Face detection"),
            "ArcFace": ("recognition", "ArcFace", "ArcFace embedding network", "Face recognition / embedding"),
            "MiniFASNet": ("liveness", "MiniFASNet", "MiniFASNet", "Passive liveness / anti-spoofing"),
        }
        for name, (stage, cache_name, model_type, purpose) in definitions.items():
            model = self._cache.get((stage, cache_name))
            if model is None:
                continue
            info = {"model_type": model_type, "purpose": purpose, "device": self.metrics.device}
            if name == "MiniFASNet":
                info.update(model.telemetry_info())
            else:
                session_model = getattr(model, "_detector", None)
                if session_model is None:
                    session_model = getattr(model, "_rec_model", None)
                session = getattr(session_model, "session", None)
                if session is not None:
                    info["framework"] = "InsightFace / ONNX Runtime"
                    info["input_dimensions"] = _session_shape(session.get_inputs()[0].shape)
                    info["output_dimensions"] = _session_shape(session.get_outputs()[0].shape)
            details[name] = info
        return details

    def select(self, *, detection: str | None = None, liveness: str | None = None, recognition: str | None = None) -> dict[str, object]:
        requested = {"detection": detection, "liveness": liveness, "recognition": recognition}
        valid = {"detection": DETECTION_MODELS, "liveness": LIVENESS_MODELS, "recognition": RECOGNITION_MODELS}
        with self._lock:
            for stage, name in requested.items():
                if name is None:
                    continue
                if name not in valid[stage]:
                    raise ValueError(f"Unsupported {stage} model: {name}")
                self._get(stage, name)  # load/validate before changing state
            self._active.update({stage: name for stage, name in requested.items() if name is not None})
            return self.status()

    def snapshot(self) -> ActiveModels:
        with self._lock:
            return ActiveModels(
                detection=self._active["detection"], liveness=self._active["liveness"], recognition=self._active["recognition"],
                detector=self._get("detection", self._active["detection"]),
                liveness_detector=self._get("liveness", self._active["liveness"]),
                recognizer=self._get("recognition", self._active["recognition"]),
            )

    def _available(self, stage: str, names: tuple[str, ...]) -> list[dict[str, object]]:
        return [{"name": name, "available": self._is_available(stage, name)} for name in names]

    def _is_available(self, stage: str, name: str) -> bool:
        if (stage, name) in self._cache:
            return True
        if stage == "liveness" and name == "CDCN":
            # CDCN is intentionally never marked ready until its calibrated
            # artifact is present. Unlike the published MiniFASNet exports,
            # it has no canonical binary ONNX release.
            return bool(settings.CDCN_MODEL_PATH and os.path.isfile(settings.CDCN_MODEL_PATH))
        if stage == "detection" and name == "MTCNN":
            return _module_available("facenet_pytorch")
        if stage == "detection" and name == "RetinaFace":
            return _module_available("retinaface")
        if stage == "recognition" and name in {"FaceNet", "VGGFace2"}:
            return _module_available("facenet_pytorch")
        if stage == "recognition" and name == "DeepFace":
            return _module_available("deepface")
        return False

    def _get(self, stage: str, name: str):
        cached = self._cache.get((stage, name))
        if cached is not None:
            return cached
        try:
            with self.metrics.measure_load(name, stage=stage):
                if stage == "detection" and name == "RetinaFace":
                    model = RetinaFaceDetector()
                elif stage == "detection" and name == "MTCNN":
                    model = MTCNNFaceDetector()
                elif stage == "recognition" and name == "FaceNet":
                    model = FaceNetRecognizer("casia-webface")
                elif stage == "recognition" and name == "VGGFace2":
                    model = FaceNetRecognizer("vggface2")
                elif stage == "recognition" and name == "DeepFace":
                    model = DeepFaceRecognizer()
                elif stage == "liveness" and name == "CDCN":
                    if not settings.CDCN_MODEL_PATH or not os.path.isfile(settings.CDCN_MODEL_PATH):
                        raise ModelUnavailableError(
                            "CDCN requires a calibrated binary ONNX model at CDCN_MODEL_PATH. "
                            "The official CDCN project ships research checkpoints, not a portable inference artifact."
                        )
                    model = LivenessDetector(
                        model_paths=settings.CDCN_MODEL_PATH,
                        model_scales=str(settings.CDCN_CROP_SCALE),
                        real_class_index=settings.CDCN_REAL_CLASS_INDEX,
                        threshold=settings.CDCN_THRESHOLD,
                        model_label="CDCN",
                    )
                else:
                    raise ModelUnavailableError(f"Unsupported {stage} model: {name}")
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not initialize {name}: {exc}. Run scripts/download_optional_models.sh first. "
                "The active model was left unchanged."
            ) from exc
        self._cache[(stage, name)] = model
        return model


def _module_available(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def _session_shape(shape: object) -> str:
    return " × ".join(str(value) for value in shape) if isinstance(shape, (list, tuple)) else str(shape)
