#!/usr/bin/env bash
set -euo pipefail

# Downloads only the optional detector/recognition weights. SCRFD, ArcFace,
# and MiniFASNet are deliberately omitted because this project already owns
# them. CDCN is omitted because its official release does not publish a
# calibrated binary ONNX artifact; see models/cdcn/README.md.
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
model_root="$script_dir/../models"
export TORCH_HOME="$model_root/torch"
export KERAS_HOME="$model_root/keras"
export DEEPFACE_HOME="$model_root/deepface"
mkdir -p "$TORCH_HOME" "$KERAS_HOME" "$model_root/retinaface" "$DEEPFACE_HOME"

python - <<'PY'
import os
import numpy as np

from facenet_pytorch import InceptionResnetV1, MTCNN

# MTCNN downloads its PNet/RNet/ONet weights.  The recognition choices use
# two distinct checkpoints: FaceNet/CASIA-WebFace and FaceNet/VGGFace2.
MTCNN(keep_all=True, device="cpu")
InceptionResnetV1(pretrained="casia-webface").eval()
InceptionResnetV1(pretrained="vggface2").eval()

# RetinaFace and DeepFace use Keras' cache, redirected above. Calling one
# inference/build pass makes their lazy downloads explicit and repeatable.
from retinaface import RetinaFace
RetinaFace.detect_faces(np.zeros((128, 128, 3), dtype=np.uint8))
from deepface import DeepFace
DeepFace.build_model("DeepFace")

print("Optional MTCNN, RetinaFace, FaceNet, VGGFace2 and DeepFace weights are ready.")
print("TORCH_HOME=", os.environ["TORCH_HOME"])
print("KERAS_HOME=", os.environ["KERAS_HOME"])
print("DEEPFACE_HOME=", os.environ["DEEPFACE_HOME"])
PY
