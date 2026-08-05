"""Shared ONNX Runtime device selection for every inference model."""
import ctypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import onnxruntime as ort

from app.config import settings


class GpuUnavailableError(RuntimeError):
    """Raised when CUDA was explicitly required but is not usable."""


@dataclass(frozen=True)
class InferenceRuntime:
    providers: list
    ctx_id: int
    device: str


def assert_cuda_session(session: object, model_name: str) -> None:
    """Ensure a mandatory-CUDA model did not silently fall back to CPU."""
    if settings.INFERENCE_DEVICE.strip().lower() != "cuda":
        return
    providers = session.get_providers()
    if "CUDAExecutionProvider" not in providers:
        raise GpuUnavailableError(
            f"{model_name} was initialized without CUDA. Active providers: {providers or 'none'}. "
            "Check NVIDIA Container Toolkit, host driver, and CUDA library configuration."
        )


def _cuda_provider_is_usable() -> bool:
    """Return whether the CUDA provider and its shared libraries can load.

    ``get_available_providers`` only reports providers compiled into the
    wheel.  It still includes CUDA when a Docker host has neither an NVIDIA
    driver nor the CUDA libraries needed by that provider.  Loading the
    provider library prevents choosing CUDA and subsequently falling back to
    CPU while incorrectly reporting a CUDA device.
    """
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False

    provider_library = Path(ort.__file__).parent / "capi" / "libonnxruntime_providers_cuda.so"
    try:
        ctypes.CDLL(str(provider_library))
    except OSError:
        return False
    return True


@lru_cache(maxsize=1)
def get_inference_runtime() -> InferenceRuntime:
    """Select one consistent execution provider for SCRFD, ArcFace, and FAS."""
    requested = settings.INFERENCE_DEVICE.strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("INFERENCE_DEVICE must be one of: auto, cuda, cpu")

    cuda_available = _cuda_provider_is_usable()
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
            ],
            ctx_id=settings.DETECTION_CTX_ID,
            device=f"cuda:{settings.DETECTION_CTX_ID}",
        )

    return InferenceRuntime(
        providers=["CPUExecutionProvider"],
        ctx_id=-1,
        device="cpu",
    )
