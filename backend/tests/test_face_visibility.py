import unittest

import numpy as np

from app.ml.face_detector import DetectedFace
from app.ml.face_visibility import FaceVisibilityEstimator


def _face(bbox=(100, 80, 300, 340)):
    return DetectedFace(
        bbox=np.asarray(bbox, dtype=np.float32),
        landmarks=np.asarray(
            [[140, 160], [260, 160], [200, 220], [160, 285], [240, 285]],
            dtype=np.float32,
        ),
        det_score=0.99,
    )


class FaceVisibilityTests(unittest.TestCase):
    def test_face_cropped_by_the_camera_edge_fails_visibility(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        assessment = FaceVisibilityEstimator().assess(image, _face(bbox=(-100, 80, 300, 340)))

        self.assertEqual(assessment.score, 0.0)

    def test_missing_eye_evidence_reduces_score_below_default_threshold(self):
        # Full face is dark except for the left-eye support patch, modelling a
        # hand or cloth hiding that eye.  Its 0.25 feature weight must make a
        # 0.80 threshold reject the frame.
        image = np.full((480, 640, 3), 30, dtype=np.uint8)
        image[134:187, 114:167] = 255
        assessment = FaceVisibilityEstimator().assess(image, _face())

        self.assertLess(assessment.score, 0.80)
        self.assertEqual(assessment.feature_scores["feature_evidence"], 0.75)

    def test_collapsed_landmarks_fail_even_when_box_is_in_frame(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        face = _face()
        face.landmarks[:] = np.asarray([[200, 200]] * 5, dtype=np.float32)

        assessment = FaceVisibilityEstimator().assess(image, face)

        self.assertEqual(assessment.score, 0.0)
