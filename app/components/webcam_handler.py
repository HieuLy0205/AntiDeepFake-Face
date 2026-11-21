"""Quản lý luồng webcam và bộ đệm trượt N frame (DD1 & Mục II.6)."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
import numpy as np

from app.components.constants import DEFAULT_MAX_FRAMES, UI_ERRORS

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


@dataclass
class WebcamCaptureResult:
    """Kết quả đọc frame từ webcam theo chuẩn RGB uint8."""

    success: bool
    frames_rgb: List[np.ndarray]
    error_message: Optional[str] = None


class SlidingFrameBuffer:
    """Bộ đệm trượt giữ tối đa N frame gần nhất cho chế độ webcam (DD1)."""

    def __init__(self, max_frames: int = DEFAULT_MAX_FRAMES) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames phải lớn hơn 0.")
        self.max_frames = max_frames
        self._buffer: Deque[np.ndarray] = deque(maxlen=max_frames)

    def push(self, frame_rgb: np.ndarray) -> None:
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("Frame đẩy vào bộ đệm phải có shape (H, W, 3) hệ RGB.")
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)
        self._buffer.append(frame_rgb)

    def get_frames(self) -> List[np.ndarray]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


def check_webcam_available(device_id: int = 0) -> Tuple[bool, Optional[str]]:
    """Kiểm tra quyền và khả năng truy cập webcam qua OpenCV."""
    if cv2 is None:
        return False, UI_ERRORS["webcam_unavailable"]

    cap = cv2.VideoCapture(device_id)
    try:
        if not cap.isOpened():
            return False, UI_ERRORS["webcam_unavailable"]
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, UI_ERRORS["webcam_unavailable"]
        return True, None
    except Exception:
        return False, UI_ERRORS["webcam_unavailable"]
    finally:
        cap.release()


def capture_webcam_burst(
    device_id: int = 0,
    num_frames: int = 5,
) -> WebcamCaptureResult:
    """Chụp chuỗi N frame RGB uint8 từ webcam cục bộ để đưa vào pipeline."""
    if cv2 is None:
        return WebcamCaptureResult(
            success=False,
            frames_rgb=[],
            error_message=UI_ERRORS["webcam_unavailable"],
        )

    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        cap.release()
        return WebcamCaptureResult(
            success=False,
            frames_rgb=[],
            error_message=UI_ERRORS["webcam_unavailable"],
        )

    buffer = SlidingFrameBuffer(max_frames=max(1, num_frames))
    try:
        for _ in range(max(1, num_frames)):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            buffer.push(frame_rgb)
    finally:
        cap.release()

    frames = buffer.get_frames()
    if not frames:
        return WebcamCaptureResult(
            success=False,
            frames_rgb=[],
            error_message=UI_ERRORS["webcam_unavailable"],
        )
    return WebcamCaptureResult(success=True, frames_rgb=frames)
