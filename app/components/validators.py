"""Kiểm tra hợp lệ đầu vào (ảnh, video, webcam) theo Mục II.6."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from app.components.constants import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_IMAGE_SIZE_MB,
    MAX_VIDEO_DURATION_SEC,
    MAX_VIDEO_SIZE_MB,
    UI_ERRORS,
)

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


@dataclass
class ValidationResult:
    """Kết quả kiểm tra đầu vào cho giao diện."""

    is_valid: bool
    error_message: Optional[str] = None
    media_type: Optional[str] = None  # "image" | "video" | "webcam"
    duration_sec: Optional[float] = None
    size_mb: float = 0.0
    width: Optional[int] = None
    height: Optional[int] = None


def _check_image_bytes(raw_bytes: bytes) -> Optional[Tuple[int, int]]:
    """Giải mã nhanh ảnh từ bytes để phát hiện tệp hỏng/giả mạo đuôi."""
    if not raw_bytes:
        return None
    try:
        from PIL import Image
        import io

        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def validate_image_input(
    filename: str,
    raw_bytes: bytes,
    max_size_mb: float = MAX_IMAGE_SIZE_MB,
) -> ValidationResult:
    """Kiểm tra tệp ảnh tải lên (JPG/PNG, không hỏng, không vượt giới hạn)."""
    ext = Path(filename or "").suffix.lower()
    size_mb = len(raw_bytes) / (1024 * 1024)

    if ext not in ALLOWED_IMAGE_EXTENSIONS or not raw_bytes:
        return ValidationResult(
            is_valid=False,
            error_message=UI_ERRORS["invalid_file"],
            media_type="image",
            size_mb=size_mb,
        )

    if size_mb > max_size_mb:
        return ValidationResult(
            is_valid=False,
            error_message=UI_ERRORS["invalid_file"],
            media_type="image",
            size_mb=size_mb,
        )

    dims = _check_image_bytes(raw_bytes)
    if dims is None:
        return ValidationResult(
            is_valid=False,
            error_message=UI_ERRORS["invalid_file"],
            media_type="image",
            size_mb=size_mb,
        )

    return ValidationResult(
        is_valid=True,
        media_type="image",
        size_mb=round(size_mb, 3),
        width=dims[0],
        height=dims[1],
    )


def probe_video_metadata(video_path: str) -> Optional[Tuple[float, int, int]]:
    """Đọc (duration_sec, width, height) của video bằng OpenCV; trả None nếu tệp hỏng."""
    if cv2 is None:
        return (0.0, 0, 0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            return None

        if fps > 0 and frame_count > 0:
            duration_sec = frame_count / fps
        else:
            duration_sec = 0.0

        return (duration_sec, width, height)
    finally:
        cap.release()


def validate_video_input(
    filename: str,
    size_bytes: int,
    video_path: Optional[str] = None,
    duration_sec_override: Optional[float] = None,
    max_size_mb: float = MAX_VIDEO_SIZE_MB,
    max_duration_sec: float = MAX_VIDEO_DURATION_SEC,
) -> ValidationResult:
    """Kiểm tra tệp video (định dạng, khả năng đọc, kích thước và độ dài)."""
    ext = Path(filename or "").suffix.lower()
    size_mb = size_bytes / (1024 * 1024)

    if ext not in ALLOWED_VIDEO_EXTENSIONS or size_bytes <= 0:
        return ValidationResult(
            is_valid=False,
            error_message=UI_ERRORS["invalid_file"],
            media_type="video",
            size_mb=size_mb,
        )

    if size_mb > max_size_mb:
        return ValidationResult(
            is_valid=False,
            error_message=UI_ERRORS["video_too_large_or_long"],
            media_type="video",
            size_mb=round(size_mb, 3),
        )

    duration_sec = duration_sec_override
    width: Optional[int] = None
    height: Optional[int] = None

    if video_path is not None:
        probed = probe_video_metadata(video_path)
        if probed is None:
            return ValidationResult(
                is_valid=False,
                error_message=UI_ERRORS["invalid_file"],
                media_type="video",
                size_mb=round(size_mb, 3),
            )
        probed_duration, width, height = probed
        if duration_sec is None:
            duration_sec = probed_duration

    if duration_sec is not None and duration_sec > max_duration_sec:
        return ValidationResult(
            is_valid=False,
            error_message=UI_ERRORS["video_too_large_or_long"],
            media_type="video",
            duration_sec=round(duration_sec, 2),
            size_mb=round(size_mb, 3),
            width=width,
            height=height,
        )

    return ValidationResult(
        is_valid=True,
        media_type="video",
        duration_sec=round(duration_sec, 2) if duration_sec is not None else None,
        size_mb=round(size_mb, 3),
        width=width,
        height=height,
    )
