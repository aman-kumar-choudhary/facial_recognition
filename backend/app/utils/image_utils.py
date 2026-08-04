import base64
import binascii
import logging
from pathlib import Path

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class InvalidImageError(ValueError):
    pass


def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode a base64 string (optionally with a data-URI prefix) into a
    BGR numpy array suitable for OpenCV / insightface pipelines."""
    if "," in image_base64 and image_base64.strip().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError(f"Could not decode base64 payload: {exc}") from exc

    max_bytes = settings.MAX_UPLOAD_IMAGE_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise InvalidImageError(
            f"Image exceeds max allowed size of {settings.MAX_UPLOAD_IMAGE_MB} MB"
        )

    buffer = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("Payload is not a valid image (decode failed)")
    return img


def save_registration_images(student_id: str, images: dict[str, np.ndarray]) -> list[Path]:
    """Persist only the configured enrollment image representation(s)."""
    directory = Path(settings.STUDENT_IMAGE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        for pose, image_bgr in images.items():
            if settings.IMAGE_STORAGE_MODE in ("color", "both"):
                color_path = directory / f"{student_id}_{pose}.jpg"
                if not cv2.imwrite(str(color_path), image_bgr):
                    raise OSError(f"Could not save registration pose '{pose}'")
                paths.append(color_path)
            if settings.IMAGE_STORAGE_MODE in ("grayscale", "both"):
                grayscale_path = directory / f"{student_id}_{pose}_gray.jpg"
                grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                if not cv2.imwrite(str(grayscale_path), grayscale):
                    raise OSError(f"Could not save registration pose '{pose}'")
                paths.append(grayscale_path)
    except Exception:
        for path in paths:
            path.unlink(missing_ok=True)
        raise
    logger.info("registration_images_saved", extra={"event": "registration_images_saved", "student_id": student_id, "pose_count": len(images), "paths": [str(path) for path in paths]})
    return paths
