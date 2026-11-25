"""Self-contained image preprocessing for Stage 2 Deepfake models (M5).

Receives RGB `np.ndarray` (H, W, 3) uint8 from M3 (`FaceCropResult.face_rgb`)
and encapsulates all model-specific resizing/normalization without mutating input.
"""

from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np
from PIL import Image


DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
HALF_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
HALF_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)


def validate_rgb_face(face_rgb: np.ndarray) -> np.ndarray:
    """Validate and coerce an input face crop to contiguous (H, W, 3) RGB uint8."""
    if not isinstance(face_rgb, np.ndarray):
        raise TypeError(f"Expected face_rgb as np.ndarray, got {type(face_rgb).__name__}")
    if face_rgb.ndim != 3 or face_rgb.shape[2] != 3:
        raise ValueError(f"Expected face_rgb shape (H, W, 3), got {face_rgb.shape}")
    if face_rgb.size == 0:
        raise ValueError("face_rgb is empty")
    if face_rgb.dtype != np.uint8:
        arr = np.clip(face_rgb, 0, 255).astype(np.uint8)
    else:
        arr = face_rgb
    return np.ascontiguousarray(arr)


def face_rgb_to_pil(face_rgb: np.ndarray) -> Image.Image:
    """Convert validated (H, W, 3) RGB uint8 numpy array to a PIL RGB Image."""
    arr = validate_rgb_face(face_rgb)
    return Image.fromarray(arr, mode="RGB")


def preprocess_for_onnx(
    faces_rgb: Sequence[np.ndarray],
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: Tuple[float, float, float] = HALF_MEAN,
    std: Tuple[float, float, float] = HALF_STD,
) -> np.ndarray:
    """Convert a batch of RGB uint8 face crops into NCHW float32 array for ONNX Runtime."""
    if not faces_rgb:
        raise ValueError("faces_rgb must contain at least one image")

    target_w, target_h = int(image_size[0]), int(image_size[1])
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)

    batch_list = []
    for face in faces_rgb:
        pil_img = face_rgb_to_pil(face)
        if pil_img.size != (target_w, target_h):
            pil_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0  # (H, W, 3)
        chw = np.transpose(arr, (2, 0, 1))  # (3, H, W)
        batch_list.append(chw)

    batch = np.stack(batch_list, axis=0)  # (N, 3, H, W)
    return np.ascontiguousarray((batch - mean_arr) / std_arr, dtype=np.float32)
