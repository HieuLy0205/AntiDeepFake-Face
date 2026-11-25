"""Face detection & context-aware cropping module (M3 contract)."""

from src.face_detection.detector import crop_with_scale, detect_and_crop

__all__ = ["crop_with_scale", "detect_and_crop"]
