import os

import cv2
import numpy as np

from app.config import settings
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


class FaceNetRecognizer:
    """Inception-ResNet-v1 FaceNet embedding backend.

    ``casia-webface`` is exposed as FaceNet and ``vggface2`` as VGGFace2.
    Both produce normalized 512-dimensional embeddings, but intentionally use
    separate FAISS namespaces because their score distributions differ.
    """

    def __init__(self, pretrained: str):
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("FaceNet/VGGFace2 requires facenet-pytorch; run scripts/download_optional_models.sh.") from exc
        os.environ.setdefault("TORCH_HOME", str(settings.model_root_path / "torch"))
        self._torch = torch
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = InceptionResnetV1(pretrained=pretrained).eval().to(self._device)

    def get_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        # facenet-pytorch expects RGB, 160x160 pixels normalized to [-1, 1].
        rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
        tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).float().div(127.5).sub(1.0).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            embedding = self._model(tensor).detach().cpu().numpy().reshape(-1)
        return l2_normalize(embedding.astype(np.float32))


class DeepFaceRecognizer:
    """DeepFace's original embedding network, not the DeepFace framework API."""

    def __init__(self):
        try:
            from deepface import DeepFace
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DeepFace requires deepface; run scripts/download_optional_models.sh.") from exc
        self._deepface = DeepFace

    def get_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        # ``skip`` prevents a second detector pass; this pipeline has already
        # selected exactly one face and performed landmark alignment.
        result = self._deepface.represent(
            img_path=aligned_face_bgr,
            model_name="DeepFace",
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
        return l2_normalize(np.asarray(result[0]["embedding"], dtype=np.float32))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
