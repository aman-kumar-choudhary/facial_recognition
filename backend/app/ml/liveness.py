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
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from app.config import settings
from app.ml.runtime import assert_cuda_session, get_inference_runtime


class LivenessConfigurationError(RuntimeError):
    """Raised at startup when passive-liveness protection is unusable."""


@dataclass(frozen=True)
class _MiniFASNetModel:
    session: object
    input_name: str
    input_width: int
    input_height: int
    class_count: int
    crop_scale: float
    path: Path


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


logger = logging.getLogger(__name__)


class LivenessDetector:
    """Fuses one or more compatible Silent-Face ONNX classifiers.

    MiniFASNetV2 uses a 2.7x detector crop and MiniFASNetV1SE uses a 4.0x
    crop.  The models are fused by averaging their logits and making one
    three-class decision, which is the intended ensemble behaviour. Taking
    the minimum of independently-softmaxed probabilities incorrectly makes
    the less confident model a hard veto and causes genuine faces to be
    rejected under ordinary camera/domain variation.
    """

    def __init__(
        self,
        *,
        model_paths: str | None = None,
        model_scales: str | None = None,
        real_class_index: int | None = None,
        threshold: float | None = None,
        model_label: str = "Silent-Face",
    ):
        import onnxruntime as ort

        runtime = get_inference_runtime()

        configured_paths = settings.LIVENESS_MODEL_PATHS if model_paths is None else model_paths
        configured_scales = settings.LIVENESS_MODEL_SCALES if model_scales is None else model_scales
        paths = _csv(configured_paths)
        if not paths and model_paths is None and settings.LIVENESS_MODEL_PATH:
            paths = [settings.LIVENESS_MODEL_PATH]

        scales = [float(value) for value in _csv(configured_scales)]
        self._real_class_index = settings.LIVENESS_REAL_CLASS_INDEX if real_class_index is None else real_class_index
        self._threshold = settings.LIVENESS_THRESHOLD if threshold is None else threshold
        self._model_label = model_label
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
        if not 0.0 < self._threshold < 1.0:
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
            assert_cuda_session(session, f"liveness model {model_path.name}")
            input_cfg = session.get_inputs()[0]
            input_shape = input_cfg.shape
            if len(input_shape) != 4 or input_shape[1] != 3:
                raise LivenessConfigurationError(
                    f"{model_path} must have an NCHW 3-channel image input; got {input_shape}."
                )
            input_height, input_width = input_shape[2], input_shape[3]
            if not isinstance(input_height, int) or not isinstance(input_width, int):
                raise LivenessConfigurationError(f"{model_path} must use a fixed image size.")
            output_shape = session.get_outputs()[0].shape
            if len(output_shape) != 2 or not isinstance(output_shape[1], int):
                raise LivenessConfigurationError(
                    f"{model_path} must have a fixed [batch, classes] output; got {output_shape}."
                )
            if self._real_class_index >= output_shape[1]:
                raise LivenessConfigurationError(
                    f"real class index is invalid for {model_path}; output has {output_shape[1]} classes."
                )
            self._models.append(
                _MiniFASNetModel(
                    session=session,
                    input_name=input_cfg.name,
                    input_width=input_width,
                    input_height=input_height,
                    class_count=output_shape[1],
                    crop_scale=crop_scale,
                    path=model_path,
                )
            )

    @property
    def model_count(self) -> int:
        return len(self._models)

    def telemetry_info(self) -> dict[str, object]:
        """Facts exposed by the loaded ONNX liveness session(s)."""
        inputs = sorted({f"{model.input_width}×{model.input_height}" for model in self._models})
        classes = sorted({model.class_count for model in self._models})
        return {
            "model_type": self._model_label,
            "purpose": "Passive liveness / anti-spoofing",
            "framework": "ONNX Runtime",
            "input_dimensions": ", ".join(inputs),
            "output_dimensions": f"{classes[0]} classes" if len(classes) == 1 else ", ".join(f"{count} classes" for count in classes),
        }

    def predict(self, image_bgr: np.ndarray, bbox_xyxy: Sequence[float]) -> tuple[bool, float]:
        """Return whether a detector face is live and its fused live score."""
        logits = [self._predict_model(model, image_bgr, bbox_xyxy) for model in self._models]
        if len({model.class_count for model in self._models}) != 1:
            raise LivenessConfigurationError("All configured liveness models must have the same class count.")
        fused_probabilities = self._softmax(np.mean(np.vstack(logits), axis=0))
        real_index = self._real_class_index
        if real_index >= fused_probabilities.size:
            raise LivenessConfigurationError(
                f"LIVENESS_REAL_CLASS_INDEX is invalid; output has {fused_probabilities.size} classes."
            )
        live_score = float(fused_probabilities[real_index])
        # Confidence alone is insufficient: a spoof class must never be
        # accepted merely because it has a non-trivial real probability.
        is_live = int(np.argmax(fused_probabilities)) == real_index and live_score >= self._threshold
        logger.debug(
            "liveness_assessed",
            extra={
                "event": "liveness_assessed",
                "model_real_scores": [round(float(self._softmax(value)[real_index]), 4) for value in logits],
                "fused_probabilities": [round(float(value), 4) for value in fused_probabilities],
                "predicted_class": int(np.argmax(fused_probabilities)),
                "real_class_index": real_index,
                "threshold": self._threshold,
                "is_live": is_live,
            },
        )
        return is_live, live_score

    def _predict_model(
        self,
        model: _MiniFASNetModel,
        image_bgr: np.ndarray,
        bbox_xyxy: Sequence[float],
    ) -> np.ndarray:
        crop = self._expanded_face_crop(image_bgr, bbox_xyxy, model.crop_scale)
        resized = cv2.resize(crop, (model.input_width, model.input_height), interpolation=cv2.INTER_LINEAR)
        # Silent-Face's exported MiniFASNet weights consume BGR pixels in [0, 255].
        tensor = np.ascontiguousarray(np.transpose(resized.astype(np.float32), (2, 0, 1))[np.newaxis, ...])
        logits = np.asarray(model.session.run(None, {model.input_name: tensor})[0])
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise LivenessConfigurationError(f"Unexpected output shape from {model.path}: {logits.shape}.")
        return np.asarray(logits[0], dtype=np.float64)

    @staticmethod
    def _expanded_face_crop(
        image_bgr: np.ndarray, bbox_xyxy: Sequence[float], scale: float
    ) -> np.ndarray:
        """Apply the crop convention used by the exported MiniFASNet models.

        The source implementation reduces the requested crop scale at an
        image edge.  Reflect-padding changed those pixels and produced a
        distribution the models were not trained to see.
        """
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Liveness input must be a BGR image with three channels.")
        image_height, image_width = image_bgr.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 1 or box_height <= 1:
            raise ValueError("Detector returned an invalid face bounding box.")

        crop_scale = min((image_height - 1) / box_height, (image_width - 1) / box_width, scale)
        crop_width, crop_height = box_width * crop_scale, box_height * crop_scale
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        left = max(0, int(center_x - crop_width / 2.0))
        top = max(0, int(center_y - crop_height / 2.0))
        right = min(image_width - 1, int(center_x + crop_width / 2.0))
        bottom = min(image_height - 1, int(center_y + crop_height / 2.0))
        if right < left or bottom < top:
            raise ValueError("Expanded face crop is empty.")
        return image_bgr[top : bottom + 1, left : right + 1]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)
