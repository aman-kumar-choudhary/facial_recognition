"""
Orchestrates the full model pipeline for both registration and
authentication, matching the flow in the architecture diagram:

Registration:  five guided captures -> SCRFD -> align -> ArcFace -> embedding
Authentication: capture -> SCRFD -> single-face check -> visibility gate -> MiniFASNet ->
                ArcFace -> embedding -> vector search -> cosine similarity

MiniFASNet runs only during authentication. Models are loaded once
(process-level singletons) since initialization is comparatively expensive;
see app/main.py.
lifespan for where these singletons are created.
"""
import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.ml.face_detector import FaceDetector, DetectedFace
from app.config import settings
from app.ml.face_align import align_face
from app.ml.face_recognizer import FaceRecognizer
from app.ml.liveness import LivenessDetector
from app.ml.face_visibility import FaceVisibilityEstimator
from app.ml.model_manager import ModelManager


@dataclass
class RegistrationResult:
    embeddings: dict[str, np.ndarray]
    quality_score: float
    timings: dict[str, float]
    models: dict[str, str]


@dataclass
class LivenessRejected(Exception):
    score: float
    timings: dict[str, float] | None = None


@dataclass
class FaceVisibilityRejected(Exception):
    score: float
    threshold: float
    timings: dict[str, float] | None = None
    feature_scores: dict[str, float] | None = None


@dataclass(frozen=True)
class PositionAssessment:
    state: str
    message: str
    face_count: int


logger = logging.getLogger(__name__)


class FacePipeline:
    def __init__(self, model_manager: ModelManager):
        self._models = model_manager
        self._visibility = FaceVisibilityEstimator()

    def process_for_registration(self, image_bgr: np.ndarray) -> RegistrationResult:
        started = time.perf_counter()
        models = self._models.snapshot()
        timings: dict[str, float] = {}
        with self._models.metrics.measure_inference(models.detection, "detection") as measurement:
            face = models.detector.detect_single(image_bgr)
        timings["detection"] = measurement.latency_ms or 0.0
        # Enrollment captures are user-guided and must not be blocked by the
        # passive anti-spoof model. MiniFASNet is deliberately applied only
        # during verification, where it protects the identity decision.
        stage_started = time.perf_counter()
        color_aligned, grayscale_aligned = self._aligned_variants(image_bgr, face)
        timings["alignment"] = self._elapsed(stage_started)
        stage_started = time.perf_counter()
        embeddings = self._base_embeddings(models.recognizer, color_aligned, grayscale_aligned, models.recognition)
        timings["recognition"] = self._elapsed(stage_started)
        logger.info(
            "registration_embeddings_generated",
            extra={
                "event": "registration_embeddings_generated",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "step_latencies_ms": timings,
            },
        )
        return RegistrationResult(
            embeddings=embeddings,
            quality_score=face.det_score,
            timings=timings,
            models={"detection": models.detection, "liveness": models.liveness, "recognition": models.recognition},
        )

    def process_for_authentication(self, image_bgr: np.ndarray) -> tuple[dict[str, np.ndarray], DetectedFace, float, float, dict[str, float], dict[str, str]]:
        """Returns configured embeddings, detected_face, and liveness score.
        Raises NoFaceDetectedError / MultipleFacesDetectedError from
        face_detector, FaceVisibilityRejected for an occluded face, or
        LivenessRejected if the face fails liveness."""
        started = time.perf_counter()
        models = self._models.snapshot()
        timings: dict[str, float] = {}
        with self._models.metrics.measure_inference(models.detection, "detection") as measurement:
            face = models.detector.detect_single(image_bgr)
        timings["detection"] = measurement.latency_ms or 0.0

        stage_started = time.perf_counter()
        visibility = self._visibility.assess(image_bgr, face)
        timings["visibility"] = self._elapsed(stage_started)
        visibility_accepted = visibility.score >= settings.FACE_VISIBILITY_THRESHOLD
        logger.info(
            "face_visibility_assessed",
            extra={
                "event": "face_visibility_assessed",
                "score": visibility.score,
                "threshold": settings.FACE_VISIBILITY_THRESHOLD,
                "accepted": visibility_accepted,
                "feature_scores": visibility.feature_scores,
            },
        )
        if not visibility_accepted:
            raise FaceVisibilityRejected(
                visibility.score,
                settings.FACE_VISIBILITY_THRESHOLD,
                timings,
                visibility.feature_scores,
            )

        # Use exactly the SCRFD detection that will be aligned for ArcFace.
        # Silent-Face needs the original frame plus this bbox, not the aligned
        # 112x112 ArcFace image or a tightly-clipped face.
        with self._models.metrics.measure_inference(models.liveness, "liveness") as measurement:
            is_live, liveness_score = models.liveness_detector.predict(image_bgr, face.bbox)
        timings["liveness"] = measurement.latency_ms or 0.0
        if not is_live:
            raise LivenessRejected(liveness_score, timings)

        stage_started = time.perf_counter()
        color_aligned, grayscale_aligned = self._aligned_variants(image_bgr, face)
        timings["alignment"] = self._elapsed(stage_started)
        stage_started = time.perf_counter()
        embeddings = self._base_embeddings(models.recognizer, color_aligned, grayscale_aligned, models.recognition)
        timings["recognition"] = self._elapsed(stage_started)
        logger.debug(
            "authentication_embeddings_generated",
            extra={
                "event": "authentication_embeddings_generated",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "step_latencies_ms": timings,
            },
        )
        return embeddings, face, visibility.score, liveness_score, timings, {"detection": models.detection, "liveness": models.liveness, "recognition": models.recognition}

    def _aligned_variants(self, image_bgr: np.ndarray, face: DetectedFace) -> tuple[np.ndarray | None, np.ndarray | None]:
        mode = settings.IMAGE_STORAGE_MODE
        color_aligned = align_face(image_bgr, face.landmarks) if mode in ("color", "both") else None
        grayscale_aligned = None
        if mode in ("grayscale", "both"):
            grayscale_aligned = align_face(self._as_grayscale_bgr(image_bgr), face.landmarks)
        return color_aligned, grayscale_aligned

    def _base_embeddings(self, recognizer: FaceRecognizer, color_aligned: np.ndarray | None, grayscale_aligned: np.ndarray | None, model_name: str) -> dict[str, np.ndarray]:
        """Return only the configured representation(s) for one pose."""
        embeddings: dict[str, np.ndarray] = {}
        if color_aligned is not None:
            with self._models.metrics.measure_inference(model_name, "recognition"):
                embeddings["color"] = recognizer.get_embedding(color_aligned)
        if grayscale_aligned is not None:
            with self._models.metrics.measure_inference(model_name, "recognition"):
                embeddings["grayscale"] = recognizer.get_embedding(grayscale_aligned)
        return embeddings

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)

    def assess_position(self, image_bgr: np.ndarray, pose: str = "center") -> PositionAssessment:
        """Fast detection-only gate used by the live camera before capture.

        This deliberately does not run liveness or ArcFace.  The expensive
        checks still happen in ``process_for_authentication`` only after the
        face has been consistently centred in the browser viewfinder.
        """
        faces = self._models.snapshot().detector.detect(image_bgr)
        if not faces:
            return PositionAssessment("no_face", "No face detected", 0)
        if len(faces) != 1:
            return PositionAssessment("misaligned", "Keep only one face in the frame", len(faces))

        face = faces[0]
        visibility = self._visibility.assess(image_bgr, face)
        # Profile captures intentionally have asymmetric landmarks.  The
        # complete-face visibility policy applies to the frontal verification
        # capture; registration still permits its guided side poses.
        if pose == "center" and visibility.score < settings.FACE_VISIBILITY_THRESHOLD:
            return PositionAssessment("misaligned", "Show your complete face inside the oval", 1)
        height, width = image_bgr.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in face.bbox)
        face_width, face_height = x2 - x1, y2 - y1
        center_x, center_y = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        width_ratio, height_ratio = face_width / width, face_height / height
        margin = min(x1 / width, y1 / height, (width - x2) / width, (height - y2) / height)

        # A side profile intentionally has an unbalanced nose/eye geometry.
        # Only the centre capture should enforce the frontal-face constraint.
        left_eye, right_eye, nose = face.landmarks[:3]
        eye_distance = float(np.linalg.norm(right_eye - left_eye))
        roll_degrees = abs(float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))))
        nose_balance = abs(float(np.linalg.norm(nose - left_eye) - np.linalg.norm(right_eye - nose))) / max(eye_distance, 1.0)

        positioned = (
            0.18 <= width_ratio <= 0.65
            and 0.22 <= height_ratio <= 0.75
            and abs(center_x - 0.5) <= 0.18
            and abs(center_y - 0.5) <= 0.22
            and margin >= 0.01
            and roll_degrees <= (12 if pose == "center" else 24)
        )
        if pose == "center":
            positioned = positioned and nose_balance <= 0.45
        if positioned:
            return PositionAssessment("ready", "Face is positioned — ready to capture", 1)
        return PositionAssessment("misaligned", "Keep your face inside the oval", 1)

    @staticmethod
    def _as_grayscale_bgr(image_bgr: np.ndarray) -> np.ndarray:
        grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
