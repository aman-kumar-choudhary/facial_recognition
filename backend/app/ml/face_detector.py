"""Face-detector adapters with a common landmark-aware contract."""
from dataclasses import dataclass
import logging
from typing import List

import numpy as np
from app.config import settings
from app.ml.model_registry import FaceModelRegistry, get_face_model_registry

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    bbox: np.ndarray          # [x1, y1, x2, y2]
    landmarks: np.ndarray     # 5x2 keypoints (eyes, nose, mouth corners)
    det_score: float
    embedding: np.ndarray | None = None  # populated later by recognizer


class NoFaceDetectedError(Exception):
    pass


class MultipleFacesDetectedError(Exception):
    def __init__(self, count: int):
        self.count = count
        super().__init__(f"Expected exactly one face, found {count}")


class FaceDetector:
    """A lightweight wrapper around the registry's one prepared SCRFD model."""

    def __init__(self, registry: FaceModelRegistry | None = None):
        self._registry = registry or get_face_model_registry()
        self._detector = self._registry.detector

    def detect(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        bboxes, keypoints = self._detector.detect(image_bgr, max_num=0, metric="default")
        if bboxes.shape[0] == 0:
            logger.debug("face_detection_completed", extra={"event": "face_detection_completed", "face_count": 0})
            return []
        results = []
        for index, bbox_with_score in enumerate(bboxes):
            det_score = float(bbox_with_score[4])
            if det_score < settings.DETECTION_MIN_CONFIDENCE:
                continue
            if keypoints is None:
                raise RuntimeError("SCRFD detector did not return the five landmarks required for ArcFace alignment.")
            results.append(
                DetectedFace(
                    bbox=bbox_with_score[:4],
                    landmarks=keypoints[index],
                    det_score=det_score,
                )
            )
        logger.debug(
            "face_detection_completed",
            extra={"event": "face_detection_completed", "face_count": len(results)},
        )
        return results

    def detect_single(self, image_bgr: np.ndarray) -> DetectedFace:
        """Enforces the 'only one face processed at a time' business rule."""
        faces = self.detect(image_bgr)
        if len(faces) == 0:
            raise NoFaceDetectedError("No face detected in frame")
        if len(faces) > 1:
            raise MultipleFacesDetectedError(len(faces))
        return faces[0]


class MTCNNFaceDetector:
    """MTCNN from facenet-pytorch.

    Its PNet/RNet/ONet weights are downloaded by facenet-pytorch into
    ``MODEL_ROOT/torch`` (configured before model construction).
    """

    def __init__(self):
        try:
            from facenet_pytorch import MTCNN
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("MTCNN requires facenet-pytorch; run scripts/download_optional_models.sh.") from exc
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._detector = MTCNN(keep_all=True, device=device, post_process=False)

    def detect(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, probabilities, landmarks = self._detector.detect(rgb, landmarks=True)
        if boxes is None:
            return []
        faces = []
        for bbox, probability, points in zip(boxes, probabilities, landmarks):
            if probability is None or float(probability) < settings.DETECTION_MIN_CONFIDENCE:
                continue
            if points is None:
                continue
            faces.append(DetectedFace(np.asarray(bbox, dtype=np.float32), np.asarray(points, dtype=np.float32), float(probability)))
        return faces

    def detect_single(self, image_bgr: np.ndarray) -> DetectedFace:
        return _detect_single(self.detect(image_bgr))


class RetinaFaceDetector:
    """RetinaFace adapter preserving its five facial landmarks."""

    def __init__(self):
        try:
            from retinaface import RetinaFace
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("RetinaFace requires retina-face; run scripts/download_optional_models.sh.") from exc
        self._retinaface = RetinaFace

    def detect(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        detections = self._retinaface.detect_faces(image_bgr, threshold=settings.DETECTION_MIN_CONFIDENCE)
        if not isinstance(detections, dict):
            return []
        faces = []
        for result in detections.values():
            score = float(result["score"])
            area = result["facial_area"]
            landmarks = result["landmarks"]
            # RetinaFace keys are stable across its TensorFlow releases.
            points = np.asarray([
                landmarks["right_eye"], landmarks["left_eye"], landmarks["nose"],
                landmarks["mouth_right"], landmarks["mouth_left"],
            ], dtype=np.float32)
            # Our aligner expects image-left eye first, regardless of the
            # subject-facing labels emitted by a detector.
            if points[0, 0] > points[1, 0]:
                points[[0, 1]] = points[[1, 0]]
            if points[3, 0] > points[4, 0]:
                points[[3, 4]] = points[[4, 3]]
            faces.append(DetectedFace(np.asarray(area, dtype=np.float32), points, score))
        return faces

    def detect_single(self, image_bgr: np.ndarray) -> DetectedFace:
        return _detect_single(self.detect(image_bgr))


def _detect_single(faces: List[DetectedFace]) -> DetectedFace:
    if len(faces) == 0:
        raise NoFaceDetectedError("No face detected in frame")
    if len(faces) > 1:
        raise MultipleFacesDetectedError(len(faces))
    return faces[0]
