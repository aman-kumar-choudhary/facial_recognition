"""Facial-feature evidence and frame-integrity visibility gate.

SCRFD can retain a face box and five estimated landmarks while a hand, mask,
or cloth hides an eye or other facial feature.  Bounding-box geometry alone
therefore must never be treated as visibility.  The score is the weighted
portion of the expected eye/nose/mouth evidence that remains visible, reduced
further if the detector face is cropped by the camera frame.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import settings
from app.ml.face_detector import DetectedFace


@dataclass(frozen=True)
class FaceVisibilityAssessment:
    score: float
    feature_scores: dict[str, float]


class FaceVisibilityEstimator:
    """Estimate visible face evidence around the detector's five landmarks.

    The configured threshold is applied directly to ``score``. For example,
    with the normal 0.80 threshold, one blocked eye (weight 0.25) is enough
    to reject a face even if SCRFD still reports that eye's location.
    """

    _FEATURES = (
        ("left_eye", 0.25, "eye"),
        ("right_eye", 0.25, "eye"),
        ("nose", 0.20, "nose"),
        ("left_mouth", 0.15, "mouth"),
        ("right_mouth", 0.15, "mouth"),
    )

    def assess(self, image_bgr: np.ndarray, face: DetectedFace) -> FaceVisibilityAssessment:
        landmarks = np.asarray(face.landmarks, dtype=np.float32)
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or landmarks.shape != (5, 2):
            return FaceVisibilityAssessment(0.0, {})

        height, width = image_bgr.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in face.bbox)
        face_width, face_height = x2 - x1, y2 - y1
        if face_width <= 1.0 or face_height <= 1.0:
            return FaceVisibilityAssessment(0.0, {})

        clipped_width = max(0.0, min(x2, width) - max(x1, 0.0))
        clipped_height = max(0.0, min(y2, height) - max(y1, 0.0))
        box_coverage = (clipped_width * clipped_height) / (face_width * face_height)
        layout_score = self._landmark_layout_score(landmarks, face_width, face_height)
        frame_margin_score = self._frame_margin_score(x1, y1, x2, y2, width, height)
        if layout_score == 0.0:
            return FaceVisibilityAssessment(0.0, {"box_coverage": round(box_coverage, 4), "frame_margin": round(frame_margin_score, 4), "landmark_layout": 0.0})

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face_patch = gray[max(0, int(y1)) : min(height, int(np.ceil(y2))), max(0, int(x1)) : min(width, int(np.ceil(x2)))]
        if face_patch.size == 0:
            return FaceVisibilityAssessment(0.0, {})
        radius = max(5, int(min(face_width, face_height) * 0.105))
        feature_scores: dict[str, float] = {}
        visible_area = 0.0
        for (name, weight, feature_type), landmark in zip(self._FEATURES, landmarks):
            evidence = self._feature_evidence(gray, landmark, radius, feature_type)
            is_visible = evidence >= self._required_evidence(feature_type)
            feature_scores[name] = round(evidence, 4)
            if is_visible:
                visible_area += weight

        feature_score = visible_area
        # A cropped face must not become valid merely because its remaining
        # visible pixels include eyes/mouth.  A normal in-frame face retains
        # a multiplier of 1.0, so this cannot create the former false reject.
        score = feature_score * box_coverage * frame_margin_score
        feature_scores = {
            "feature_evidence": round(feature_score, 4),
            "box_coverage": round(box_coverage, 4),
            "frame_margin": round(frame_margin_score, 4),
            "landmark_layout": round(layout_score, 4),
            **feature_scores,
        }
        return FaceVisibilityAssessment(score=round(float(np.clip(score, 0.0, 1.0)), 4), feature_scores=feature_scores)

    @staticmethod
    def _feature_evidence(gray: np.ndarray, landmark: np.ndarray, radius: int, feature_type: str) -> float:
        """Measure landmark-local facial detail, independent of exposure.

        SCRFD only returns landmark coordinates, not landmark confidences. A
        covered feature can consequently retain a plausible coordinate. The
        old global dark-pixel test incorrectly accepted a dark hand, mask, or
        shadow as an eye/mouth. Here the centre is compared with its immediate
        surrounding ring: eyes and mouth need their characteristic dark
        contrast, and the nose needs local edge detail. Uniform occluders do
        not satisfy either measurement.
        """
        x, y = (int(round(value)) for value in landmark)
        outer_radius = int(radius * 1.8)
        y1, y2 = max(0, y - outer_radius), min(gray.shape[0], y + outer_radius + 1)
        x1, x2 = max(0, x - outer_radius), min(gray.shape[1], x + outer_radius + 1)
        patch = gray[y1:y2, x1:x2].astype(np.float32)
        if patch.shape[0] < outer_radius or patch.shape[1] < outer_radius:
            return 0.0
        yy, xx = np.ogrid[y1:y2, x1:x2]
        distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        centre = patch[distance <= radius]
        ring = patch[(distance >= radius * 1.25) & (distance <= outer_radius)]
        if centre.size == 0 or ring.size == 0:
            return 0.0
        if feature_type in {"eye", "mouth"}:
            # Contrast relative to the immediately adjacent skin is robust to
            # dark rooms and darker skin tones, unlike an absolute threshold.
            contrast = max(0.0, float(np.median(ring) - np.percentile(centre, 30)))
            return float(np.clip(contrast / 32.0, 0.0, 1.0))

        gradient_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        centre_detail = float(np.mean(cv2.magnitude(gradient_x, gradient_y)[distance <= radius]))
        ring_detail = float(np.mean(cv2.magnitude(gradient_x, gradient_y)[(distance >= radius * 1.25) & (distance <= outer_radius)]))
        return float(np.clip(max(0.0, centre_detail - ring_detail * 0.55) / 28.0, 0.0, 1.0))

    @staticmethod
    def _required_evidence(feature_type: str) -> float:
        return {
            # Keep the existing environment variable names for deployment
            # compatibility; they now represent normalized local evidence
            # floors rather than raw dark-pixel ratios.
            "eye": settings.FACE_VISIBILITY_EYE_DARK_PIXEL_RATIO,
            "nose": settings.FACE_VISIBILITY_NOSE_DARK_PIXEL_RATIO,
            "mouth": settings.FACE_VISIBILITY_MOUTH_DARK_PIXEL_RATIO,
        }[feature_type]

    @staticmethod
    def _frame_margin_score(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> float:
        required = min(width, height) * settings.FACE_VISIBILITY_FRAME_MARGIN_RATIO
        if required <= 0:
            return 1.0
        margin = min(x1, y1, width - x2, height - y2)
        return float(np.clip(margin / required, 0.0, 1.0))

    @staticmethod
    def _landmark_layout_score(landmarks: np.ndarray, face_width: float, face_height: float) -> float:
        left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
        eye_distance = float(np.linalg.norm(right_eye - left_eye))
        mouth_distance = float(np.linalg.norm(right_mouth - left_mouth))
        vertical_span = float(np.mean([left_mouth[1], right_mouth[1]]) - np.mean([left_eye[1], right_eye[1]]))
        min_spread = min(face_width, face_height) * settings.FACE_VISIBILITY_MIN_LANDMARK_SPREAD_RATIO

        # The order/relative vertical positions are detector-independent and
        # tolerate a modest head turn.  A collapsed set of points is not.
        coherent = (
            eye_distance >= min_spread
            and mouth_distance >= min_spread * 0.55
            and vertical_span >= min_spread
            and min(left_eye[0], right_eye[0]) <= nose[0] <= max(left_eye[0], right_eye[0])
        )
        return 1.0 if coherent else 0.0
