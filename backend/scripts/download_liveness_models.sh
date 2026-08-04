#!/usr/bin/env bash
set -euo pipefail

# Compatible ONNX exports of the public Silent-Face-Anti-Spoofing models.
# See backend/models/liveness/README.md for source and integrity hashes.
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
model_dir="$script_dir/../models/liveness"
mkdir -p "$model_dir"

download_model() {
  local filename="$1"
  local checksum="$2"
  local url="$3"
  local destination="$model_dir/$filename"
  local temporary="$destination.download"

  if test -f "$destination" && printf '%s  %s\n' "$checksum" "$filename" \
      | (cd "$model_dir" && sha256sum --check --status --strict); then
    printf 'Verified existing %s\n' "$filename"
    return
  fi

  rm -f "$temporary"
  curl --fail --location --retry 3 --output "$temporary" "$url"
  printf '%s  %s\n' "$checksum" "$temporary" | (cd "$model_dir" && sha256sum --check --strict)
  mv "$temporary" "$destination"
}

download_model \
  'MiniFASNetV2.onnx' \
  'b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907' \
  'https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx'
download_model \
  'MiniFASNetV1SE.onnx' \
  'ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676' \
  'https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx'
