"""Unified frame reader for images, short videos, and live webcams (M3 contract).

Guarantees:
- Output frames are strictly RGB `np.ndarray` with `dtype=np.uint8` and shape `(H, W, 3)`.
- Raises `InvalidInputError` on missing, empty, corrupt, or unreadable inputs.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import List, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.schemas import FramePacket


IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


class InvalidInputError(ValueError):
    """Raised when an image, video, or webcam input cannot be opened or decoded."""


def _load_single_image(path: Path) -> List[FramePacket]:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            rgb = np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))
    except Exception as exc:
        raise InvalidInputError(f"Corrupt or unreadable image file: {path}") from exc

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise InvalidInputError(f"Invalid RGB image dimensions for: {path}")
    return [FramePacket(frame_id=0, timestamp_sec=0.0, image_rgb=rgb)]


def _load_video_file(path: Path, sample_fps: int, max_frames: int) -> List[FramePacket]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise InvalidInputError(f"Cannot open video file: {path}")

    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0.0 or not np.isfinite(src_fps):
            src_fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps_step = max(1, int(round(src_fps / max(1, int(sample_fps)))))
        # Spread max_frames across full clip duration while respecting sample_fps minimum spacing
        span_step = (total_frames // max_frames) if (total_frames > max_frames * fps_step) else fps_step
        step = max(fps_step, span_step)

        packets: List[FramePacket] = []
        raw_idx = 0
        while len(packets) < max_frames:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            if raw_idx % step == 0:
                rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
                packets.append(
                    FramePacket(
                        frame_id=len(packets),
                        timestamp_sec=round(raw_idx / src_fps, 3),
                        image_rgb=rgb,
                    )
                )
            raw_idx += 1
    finally:
        cap.release()

    if not packets:
        raise InvalidInputError(f"No valid frames decoded from video: {path}")
    return packets


def _load_webcam_stream(device_id: int, sample_fps: int, max_frames: int) -> List[FramePacket]:
    cap = cv2.VideoCapture(int(device_id))
    if not cap.isOpened():
        cap.release()
        raise InvalidInputError(f"Cannot open webcam device ID {device_id}")

    packets: List[FramePacket] = []
    interval_sec = min(0.12, 1.0 / max(1, int(sample_fps)))
    t0 = time.perf_counter()
    try:
        for idx in range(max_frames):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            packets.append(
                FramePacket(
                    frame_id=idx,
                    timestamp_sec=round(time.perf_counter() - t0, 3),
                    image_rgb=rgb,
                )
            )
            if idx + 1 < max_frames and interval_sec > 0:
                time.sleep(interval_sec)
    finally:
        cap.release()

    if not packets:
        raise InvalidInputError(f"Webcam device {device_id} returned zero frames.")
    return packets


def load_frames(
    source: Union[str, Path, int],
    sample_fps: int = 5,
    max_frames: int = 30,
) -> List[FramePacket]:
    """Decode RGB uint8 frames from an image path, video path, or integer webcam ID."""
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    if sample_fps <= 0:
        sample_fps = 5

    if isinstance(source, int) and not isinstance(source, bool):
        return _load_webcam_stream(source, sample_fps=sample_fps, max_frames=max_frames)

    path = Path(str(source))
    if not path.exists() or not path.is_file():
        raise InvalidInputError(f"Input file does not exist: {path}")
    if path.stat().st_size == 0:
        raise InvalidInputError(f"Input file is empty (0 bytes): {path}")

    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return _load_single_image(path)
    if ext in VIDEO_EXTENSIONS:
        return _load_video_file(path, sample_fps=sample_fps, max_frames=max_frames)

    raise InvalidInputError(f"Unsupported file extension '{ext}' for: {path}")
