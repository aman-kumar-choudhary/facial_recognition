"""Silent-Face passive liveness using MiniFASNet ONNX models.

The model is run on an *expanded detector bounding box*, never on an ArcFace
alignment or a tight face crop.  That is the crop convention used by
Silent-Face-Anti-Spoofing and preserves the background/edge artefacts needed
to identify printed photographs and display replays.

There is deliberately no texture-heuristic fallback.  An authentication
service that starts without its liveness model must fail closed instead of
silently accepting presentation attacks.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from app.config import settings
from app.ml.runtime import get_inference_runtime


class LivenessConfigurationError(RuntimeError):
    """Raised at startup when passive-liveness protection is unusable."""


@dataclass(frozen=True)
class _MiniFASNetModel:
    session: object
    input_name: str
    input_width: int
    input_height: int
    crop_scale: float
    path: Path


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class LivenessDetector:
    """Fuses one or more compatible Silent-Face ONNX classifiers.

    MiniFASNetV2 uses a 2.7x detector crop and MiniFASNetV1SE uses a 4.0x
    crop. The final score is the lowest configured real-class probability,
    making a model's spoof decision fail the gate rather than being hidden by
    another model's confidence.
    """

    def __init__(self):
        import onnxruntime as ort

        runtime = get_inference_runtime()

        paths = _csv(settings.LIVENESS_MODEL_PATHS)
        if not paths and settings.LIVENESS_MODEL_PATH:
            paths = [settings.LIVENESS_MODEL_PATH]

        scales = [float(value) for value in _csv(settings.LIVENESS_MODEL_SCALES)]
        if not paths:
            raise LivenessConfigurationError(
                "No Silent-Face model configured. Set LIVENESS_MODEL_PATHS before starting the API."
            )
        if not scales:
            # Retains compatibility with a previous single-model configuration.
            scales = [2.7] * len(paths)
        if len(paths) != len(scales):
            raise LivenessConfigurationError(
                "LIVENESS_MODEL_PATHS and LIVENESS_MODEL_SCALES must contain the same number of values."
            )
        if not 0.0 < settings.LIVENESS_THRESHOLD < 1.0:
            raise LivenessConfigurationError("LIVENESS_THRESHOLD must be between 0 and 1.")

        self._models: list[_MiniFASNetModel] = []
        for model_path_text, crop_scale in zip(paths, scales):
            model_path = Path(model_path_text)
            if not model_path.is_file():
                raise LivenessConfigurationError(
                    f"Silent-Face model is missing: {model_path}. Run scripts/download_liveness_models.sh."
                )
            if crop_scale <= 1.0:
                raise LivenessConfigurationError("Each Silent-Face crop scale must be greater than 1.")

            session = ort.InferenceSession(str(model_path), providers=runtime.providers)
            input_cfg = session.get_inputs()[0]
            input_shape = input_cfg.shape
            if len(input_shape) != 4 or input_shape[1] != 3:
                raise LivenessConfigurationError(
                    f"{model_path} must have an NCHW 3-channel image input; got {input_shape}."
                )
            input_height, input_width = input_shape[2], input_shape[3]
            if not isinstance(input_height, int) or not isinstance(input_width, int):
                raise LivenessConfigurationError(f"{model_path} must use a fixed image size.")
            self._models.append(
                _MiniFASNetModel(
                    session=session,
                    input_name=input_cfg.name,
                    input_width=input_width,
                    input_height=input_height,
                    crop_scale=crop_scale,
                    path=model_path,
                )
            )

    @property
    def model_count(self) -> int:
        return len(self._models)

    def predict(self, image_bgr: np.ndarray, bbox_xyxy: Sequence[float]) -> tuple[bool, float]:
        """Return whether a detector face is live and its fused live score."""
        scores = [self._predict_model(model, image_bgr, bbox_xyxy) for model in self._models]
        live_score = float(min(scores))
        return live_score >= settings.LIVENESS_THRESHOLD, live_score

    def _predict_model(
        self,
        model: _MiniFASNetModel,
        image_bgr: np.ndarray,
        bbox_xyxy: Sequence[float],
    ) -> float:
        crop = self._expanded_face_crop(image_bgr, bbox_xyxy, model.crop_scale)
        resized = cv2.resize(crop, (model.input_width, model.input_height), interpolation=cv2.INTER_LINEAR)
        # Silent-Face's exported MiniFASNet weights consume BGR pixels in [0, 255].
        tensor = np.ascontiguousarray(np.transpose(resized.astype(np.float32), (2, 0, 1))[np.newaxis, ...])
        logits = np.asarray(model.session.run(None, {model.input_name: tensor})[0])
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise LivenessConfigurationError(f"Unexpected output shape from {model.path}: {logits.shape}.")
        if settings.LIVENESS_REAL_CLASS_INDEX >= logits.shape[1]:
            raise LivenessConfigurationError(
                f"LIVENESS_REAL_CLASS_INDEX is invalid for {model.path}; output has {logits.shape[1]} classes."
            )
        probabilities = self._softmax(logits[0])
        return float(probabilities[settings.LIVENESS_REAL_CLASS_INDEX])

    @staticmethod
    def _expanded_face_crop(
        image_bgr: np.ndarray, bbox_xyxy: Sequence[float], scale: float
    ) -> np.ndarray:
        """Expand around a SCRFD bbox, reflect-padding when it reaches an edge."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Liveness input must be a BGR image with three channels.")
        image_height, image_width = image_bgr.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 1 or box_height <= 1:
            raise ValueError("Detector returned an invalid face bounding box.")

        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half_width, half_height = box_width * scale / 2.0, box_height * scale / 2.0
        # Preserve the model's requested crop geometry even when a person is
        # close to the camera. Clamping changes the effective scale to a tight
        # face crop and causes false spoof scores. Reflect-padding supplies a
        # stable border before extracting the same sized expanded region.
        left = int(np.floor(center_x - half_width))
        top = int(np.floor(center_y - half_height))
        right = int(np.ceil(center_x + half_width))
        bottom = int(np.ceil(center_y + half_height))
        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - image_width)
        pad_bottom = max(0, bottom - image_height)
        if pad_left or pad_top or pad_right or pad_bottom:
            image_bgr = cv2.copyMakeBorder(
                image_bgr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
            )
            left += pad_left
            right += pad_left
            top += pad_top
            bottom += pad_top
        if right <= left or bottom <= top:
            raise ValueError("Expanded face crop is empty.")
        return image_bgr[top:bottom, left:right]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)
