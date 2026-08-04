#!/usr/bin/env bash
set -euo pipefail

# Run inside the same Python environment that starts Uvicorn.
python -m pip uninstall -y onnxruntime
python -m pip install \
  'onnxruntime-gpu==1.19.2' \
  'nvidia-cuda-runtime-cu12' \
  'nvidia-cudnn-cu12'
python - <<'PY'
import onnxruntime as ort

providers = ort.get_available_providers()
print("ONNX Runtime providers:", providers)
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("CUDAExecutionProvider is unavailable. Check the NVIDIA driver, CUDA 12.x, and cuDNN 9.x.")
PY
