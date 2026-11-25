"""MTCNN face detection, largest-face selection, and multi-scale context crop (M3).

Implements:
- `_detect_raw_boxes(image_rgb)` returning `[((x1, y1, x2, y2), confidence), ...]`
  (injectable hook verified by `tests/test_face_detection.py`).
- `detect_and_crop(packet, target_size=(224, 224))` selecting the largest-area face
  (M3 frozen rule) and producing `(224, 224, 3)` RGB `uint8` with 25% boundary context.
- `crop_with_scale(image_rgb, bbox, scale=2.7, out_size=(80, 80))` following Minivision
  Silent-Face-Anti-Spoofing's 2.7x multi-scale context crop (DD4) so Stage 1 Anti-Spoof
  preserves paper edges, screen bezels, and peripheral specular reflections.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from src.schemas import FaceCropResult, FramePacket


_MTCNN_SINGLETON: Any = None


def _get_mtcnn() -> Any:
    global _MTCNN_SINGLETON
    if _MTCNN_SINGLETON is None:
        from facenet_pytorch import MTCNN  # type: ignore

        _MTCNN_SINGLETON = MTCNN(keep_all=True, min_face_size=40, device="cpu")
    return _MTCNN_SINGLETON


def _detect_raw_boxes(
    image_rgb: np.ndarray,
    min_prob: float = 0.80,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Detect all candidate face bounding boxes `((x1, y1, x2, y2), prob)` in an RGB frame."""
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.size == 0:
        return []

    # Fast check for completely uniform/blank frames (avoids unnecessary neural pass)
    if int(image_rgb.max()) == int(image_rgb.min()):
        return []

    h, w = image_rgb.shape[:2]
    scale = 640.0 / max(h, w) if max(h, w) > 640 else 1.0
    if scale < 1.0:
        small = cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = image_rgb

    mtcnn = _get_mtcnn()
    boxes, probs = mtcnn.detect(Image.fromarray(small, mode="RGB"))
    if boxes is None or probs is None or len(boxes) == 0:
        return []

    results: List[Tuple[Tuple[int, int, int, int], float]] = []
    for box, prob in zip(boxes, probs):
        if box is None or prob is None or float(prob) < min_prob:
            continue
        x1 = max(0, min(w - 1, int(round(float(box[0]) / scale))))
        y1 = max(0, min(h - 1, int(round(float(box[1]) / scale))))
        x2 = max(x1 + 1, min(w, int(round(float(box[2]) / scale))))
        y2 = max(y1 + 1, min(h, int(round(float(box[3]) / scale))))
        if x2 > x1 and y2 > y1:
            results.append(((x1, y1, x2, y2), float(prob)))
    return results


def crop_with_scale(
    image_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    scale: float = 2.7,
    out_size: Tuple[int, int] = (80, 80),
) -> np.ndarray:
    """Expand `bbox` around its center by `scale` (clamped to frame) and resize to `out_size`.

    Matches Silent-Face-Anti-Spoofing's `CropImage._get_new_box` (`scale=2.7` on `80x80`)
    to retain peripheral physical context (paper border, phone bezel, Moire fringes).
    """
    h_img, w_img = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    eff_scale = min((h_img - 1) / box_h, min((w_img - 1) / box_w, float(scale)))
    new_w = box_w * eff_scale
    new_h = box_h * eff_scale
    cx = box_w / 2.0 + x1
    cy = box_h / 2.0 + y1

    nx1 = max(0, int(cx - new_w / 2.0))
    ny1 = max(0, int(cy - new_h / 2.0))
    nx2 = min(w_img, int(cx + new_w / 2.0) + 1)
    ny2 = min(h_img, int(cy + new_h / 2.0) + 1)

    patch = image_rgb[ny1:ny2, nx1:nx2]
    if patch.size == 0:
        patch = image_rgb
    resized = cv2.resize(patch, (int(out_size[0]), int(out_size[1])), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def detect_and_crop(
    packet: FramePacket,
    target_size: Tuple[int, int] = (224, 224),
    margin_ratio: float = 0.25,
) -> FaceCropResult:
    """Detect faces in `packet.image_rgb`, pick largest bbox, and return `(224,224,3)` RGB uint8."""
    img = packet.image_rgb
    candidates = _detect_raw_boxes(img)
    if not candidates:
        return FaceCropResult(
            frame_id=int(packet.frame_id),
            has_face=False,
            bbox=None,
            det_confidence=0.0,
            face_rgb=None,
            face_count=0,
        )

    # Pick face with largest bounding-box area (M3 frozen multi-face rule)
    best_bbox, best_conf = max(
        candidates,
        key=lambda item: max(0, item[0][2] - item[0][0]) * max(0, item[0][3] - item[0][1]),
    )

    h, w = img.shape[:2]
    x1, y1, x2, y2 = best_bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * float(margin_ratio))
    nx1 = max(0, int(round(cx - side / 2.0)))
    ny1 = max(0, int(round(cy - side / 2.0)))
    nx2 = min(w, int(round(cx + side / 2.0)))
    ny2 = min(h, int(round(cy + side / 2.0)))

    crop = img[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    resized = cv2.resize(
        crop,
        (int(target_size[0]), int(target_size[1])),
        interpolation=cv2.INTER_LINEAR,
    )
    return FaceCropResult(
        frame_id=int(packet.frame_id),
        has_face=True,
        bbox=best_bbox,
        det_confidence=round(float(best_conf), 4),
        face_rgb=np.ascontiguousarray(resized, dtype=np.uint8),
        face_count=len(candidates),
    )
