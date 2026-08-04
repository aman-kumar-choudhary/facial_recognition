"""Shared ONNX Runtime device selection for every inference model."""
from dataclasses import dataclass
from functools import lru_cache

import onnxruntime as ort

from app.config import settings


class GpuUnavailableError(RuntimeError):
    """Raised when CUDA was explicitly required but is not usable."""


@dataclass(frozen=True)
class InferenceRuntime:
    providers: list
    ctx_id: int
    device: str


@lru_cache(maxsize=1)
def get_inference_runtime() -> InferenceRuntime:
    """Select one consistent execution provider for SCRFD, ArcFace, and FAS."""
    requested = settings.INFERENCE_DEVICE.strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("INFERENCE_DEVICE must be one of: auto, cuda, cpu")

    cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
    if requested == "cuda" and not cuda_available:
        available = ", ".join(ort.get_available_providers())
        raise GpuUnavailableError(
            "CUDA was requested but ONNX Runtime has no CUDAExecutionProvider. "
            f"Available providers: {available or 'none'}. Install onnxruntime-gpu and fix NVIDIA/CUDA."
        )

    if requested != "cpu" and cuda_available:
        return InferenceRuntime(
            providers=[
                ("CUDAExecutionProvider", {"device_id": str(settings.DETECTION_CTX_ID)}),
                "CPUExecutionProvider",
            ],
            ctx_id=settings.DETECTION_CTX_ID,
            device=f"cuda:{settings.DETECTION_CTX_ID}",
        )

    return InferenceRuntime(
        providers=["CPUExecutionProvider"],
        ctx_id=-1,
        device="cpu",
    )
