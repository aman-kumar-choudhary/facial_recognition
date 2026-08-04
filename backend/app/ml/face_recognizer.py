import numpy as np

from app.ml.model_registry import FaceModelRegistry, get_face_model_registry


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


class FaceRecognizer:
    def __init__(self, registry: FaceModelRegistry | None = None):
        self._registry = registry or get_face_model_registry()
        self._rec_model = self._registry.recognizer

    def get_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        embedding = self._rec_model.get_feat(aligned_face_bgr)
        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).flatten()

        return l2_normalize(embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
