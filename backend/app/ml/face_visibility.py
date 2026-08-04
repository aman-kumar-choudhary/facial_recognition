"""Configuration-driven, local face-occlusion assessment.

SCRFD deliberately detects partially visible faces, which is desirable for
tracking but not sufficient for an identity decision.  This module checks the
five facial landmark regions in the detected face crop before liveness or
ArcFace runs.  It is intentionally conservative: a missing eye, nose, or
mouth feature lowers the estimated visible-face score and blocks recognition
when it falls below ``FACE_VISIBILITY_THRESHOLD``.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from app.ml.face_detector import DetectedFace


@dataclass(frozen=True)
class FaceVisibilityAssessment:
    score: float
    feature_scores: dict[str, float]


class FaceVisibilityEstimator:
    """Estimate feature visibility without loading another ML model.

    The estimator uses local contrast and edge detail in landmark-centred
    regions.  Those regions contain stable eye/nose/mouth structure on an
    unobstructed face; a hand, mask, or frame edge removes that structure.
    Frame clipping is included as a hard geometric visibility loss.
    """

    _FEATURE_NAMES = ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")
    _FEATURE_WEIGHTS = np.asarray((0.24, 0.24, 0.20, 0.16, 0.16), dtype=np.float32)

    def assess(self, image_bgr: np.ndarray, face: DetectedFace) -> FaceVisibilityAssessment:
        height, width = image_bgr.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in face.bbox)
        face_width, face_height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        frame_coverage = max(0.0, min(x2, width) - max(x1, 0.0)) * max(0.0, min(y2, height) - max(y1, 0.0))
        frame_coverage /= face_width * face_height

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        radius = max(5, int(min(face_width, face_height) * 0.105))
        scores: dict[str, float] = {}
        for name, landmark in zip(self._FEATURE_NAMES, face.landmarks):
            scores[name] = self._feature_structure_score(gray, landmark, radius)

        feature_score = float(np.dot(self._FEATURE_WEIGHTS, np.asarray(list(scores.values()), dtype=np.float32)))
        # A clipped box represents pixels that are certainly not visible;
        # keep the feature calculation from overstating that case.
        return FaceVisibilityAssessment(score=round(feature_score * frame_coverage, 4), feature_scores=scores)

    @staticmethod
    def _feature_structure_score(gray: np.ndarray, landmark: np.ndarray, radius: int) -> float:
        x, y = (int(round(value)) for value in landmark)
        y1, y2 = max(0, y - radius), min(gray.shape[0], y + radius + 1)
        x1, x2 = max(0, x - radius), min(gray.shape[1], x + radius + 1)
        patch = gray[y1:y2, x1:x2]
        if patch.shape[0] < radius or patch.shape[1] < radius:
            return 0.0

        # Both local contrast and edges must be present.  Normalising against
        # calibrated low-detail floors makes the score stable across lighting
        # while still failing closed on featureless occluders.
        contrast = float(np.std(patch))
        edges = cv2.Canny(patch, 45, 120)
        edge_density = float(np.count_nonzero(edges)) / edges.size
        contrast_score = min(1.0, contrast / 22.0)
        edge_score = min(1.0, edge_density / 0.095)
        return round(0.55 * contrast_score + 0.45 * edge_score, 4)
