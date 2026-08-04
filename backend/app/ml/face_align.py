"""
Face alignment: warps the detected face crop to a canonical 112x112
template using the 5-point landmarks from SCRFD, as expected by ArcFace.
"""
import cv2
import numpy as np

# Canonical ArcFace 112x112 reference landmarks
_ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Similarity-transform the face region to the canonical ArcFace template."""
    dst = _ARCFACE_TEMPLATE.copy()
    if image_size != 112:
        dst *= image_size / 112.0

    src = landmarks.astype(np.float32)
    tform = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
    if tform is None:
        raise ValueError("Could not estimate alignment transform from landmarks")

    aligned = cv2.warpAffine(image_bgr, tform, (image_size, image_size), borderValue=0.0)
    return aligned
