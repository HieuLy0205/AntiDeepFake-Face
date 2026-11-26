import os
from collections import deque
from typing import Deque, List, Union
import cv2
import numpy as np

from src.schemas import FramePacket


class InvalidInputError(ValueError):
    """Ngoại lệ khi tệp nguồn, ảnh, video hoặc webcam không hợp lệ / không đọc được."""


class SlidingFrameBuffer:
    """Bộ đệm trượt giữ tối đa `max_frames` frame RGB uint8 gần nhất (M3 deliverable)."""

    def __init__(self, max_frames: int = 30) -> None:
        if max_frames <= 0:
            raise InvalidInputError(f"max_frames phải lớn hơn 0, nhận được {max_frames}")
        self.max_frames = int(max_frames)
        self._buffer: Deque[Union[FramePacket, np.ndarray]] = deque(maxlen=self.max_frames)

    def push(self, frame: Union[FramePacket, np.ndarray]) -> None:
        """Thêm một FramePacket hoặc ảnh RGB uint8 vào bộ đệm trượt."""
        if isinstance(frame, FramePacket):
            img = frame.image_rgb
            if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
                raise InvalidInputError("FramePacket.image_rgb phải có shape (H, W, 3) hệ RGB.")
            if img.dtype != np.uint8:
                frame = FramePacket(
                    frame_id=frame.frame_id,
                    timestamp_sec=frame.timestamp_sec,
                    image_rgb=img.astype(np.uint8),
                )
            self._buffer.append(frame)
        elif isinstance(frame, np.ndarray):
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise InvalidInputError("Ảnh đẩy vào bộ đệm phải có shape (H, W, 3) hệ RGB.")
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            self._buffer.append(frame)
        else:
            raise InvalidInputError(
                f"frame phải là FramePacket hoặc np.ndarray, nhận được {type(frame)}"
            )

    def get_frames(self) -> List[Union[FramePacket, np.ndarray]]:
        """Trả về danh sách các frame hiện có theo thứ tự thời gian."""
        return list(self._buffer)

    def clear(self) -> None:
        """Xóa toàn bộ frame khỏi bộ đệm."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


def load_single_frame(image_source: Union[str, np.ndarray], frame_id: int = 0) -> FramePacket:
    """
    Tạo FramePacket từ một ảnh tĩnh hoặc mảng NumPy (RGB) phục vụ demo Streamlit/Webcam.
    """
    if isinstance(image_source, str):
        if not os.path.exists(image_source):
            raise InvalidInputError(f"Không tìm thấy file ảnh: {image_source}")
        bgr = cv2.imread(image_source)
        if bgr is None or bgr.size == 0:
            raise InvalidInputError(f"Không thể đọc file ảnh: {image_source}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_source, np.ndarray):
        if image_source.ndim != 3 or image_source.shape[2] != 3:
            raise InvalidInputError("Mảng ảnh đầu vào phải có shape (H, W, 3) hệ RGB.")
        rgb = image_source
    else:
        raise InvalidInputError(
            f"image_source phải là đường dẫn str hoặc np.ndarray, nhận được {type(image_source)}"
        )

    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)

    return FramePacket(
        frame_id=frame_id,
        timestamp_sec=0.0,
        image_rgb=rgb,
    )


def _select_scene_or_uniform_indices(
    thumbs: List[np.ndarray],
    max_frames: int,
    sampling_mode: str,
) -> List[int]:
    """Chọn tối đa `max_frames` chỉ số đại diện trải đều toàn video hoặc theo chuyển cảnh."""
    total = len(thumbs)
    if total <= max_frames:
        return list(range(total))
    if max_frames <= 1:
        return [0]

    if sampling_mode == "uniform":
        chosen: List[int] = []
        for i in range(max_frames):
            idx = int(round(i * (total - 1) / (max_frames - 1)))
            if not chosen or idx > chosen[-1]:
                chosen.append(idx)
        return chosen

    # sampling_mode == "scene": ưu tiên các bước nhảy chuyển cảnh khác biệt lớn nhất
    diffs: List[float] = [255.0]
    for k in range(1, total):
        d = float(np.mean(np.abs(thumbs[k].astype(np.float32) - thumbs[k - 1].astype(np.float32))))
        diffs.append(d)

    selected: List[int] = [0]
    min_gap = max(1, total // (max_frames * 2))
    ranked = sorted(range(1, total), key=lambda k: diffs[k], reverse=True)
    for k in ranked:
        if len(selected) >= max_frames:
            break
        if diffs[k] >= 4.0 and all(abs(k - s) >= min_gap for s in selected):
            selected.append(k)

    # Pass 2: lấp đầy các khoảng trống trên dòng thời gian để phủ đều toàn video
    selected_set = set(selected)
    while len(selected) < max_frames:
        best_idx = None
        best_score = (-1, -1.0)
        for k in range(total):
            if k in selected_set:
                continue
            nearest = min(abs(k - s) for s in selected)
            score = (nearest, diffs[k])
            if score > best_score:
                best_score = score
                best_idx = k
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)

    selected.sort()
    return selected


def load_frames(
    source: Union[str, int],
    sample_fps: int = 5,
    max_frames: int = 30,
    sampling_mode: str = "fps",
) -> List[FramePacket]:
    """
    Đọc frame từ video, ảnh hoặc webcam và chuyển về hệ màu RGB uint8.

    Args:
        source: Đường dẫn tới file ảnh/video hoặc ID webcam (int).
        sample_fps: Tần suất lấy mẫu frame (số frame trên mỗi giây video).
        max_frames: Số lượng frame tối đa cần lấy.
        sampling_mode: `"fps"` (mặc định, đều theo FPS từ đầu), `"uniform"` (trải đều toàn video),
            hoặc `"scene"` (tự động bắt các khung cảnh khác nhau trên toàn video).

    Returns:
        List[FramePacket]: Danh sách các frame theo định dạng chuẩn FramePacket (RGB uint8).
    """
    if sample_fps <= 0:
        raise InvalidInputError(f"sample_fps phải lớn hơn 0, nhận được {sample_fps}")
    if max_frames <= 0:
        raise InvalidInputError(f"max_frames phải lớn hơn 0, nhận được {max_frames}")
    if not isinstance(source, (str, int)) or isinstance(source, bool):
        raise InvalidInputError(f"Nguồn đầu vào không hợp lệ: {source!r}")

    packets: List[FramePacket] = []

    # 1. Trường hợp nguồn là file ảnh tĩnh
    if isinstance(source, str):
        if not os.path.exists(source):
            raise InvalidInputError(f"Không tìm thấy file nguồn: {source}")

        lower_path = source.lower()
        if lower_path.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            bgr = cv2.imread(source)
            if bgr is None or bgr.size == 0:
                raise InvalidInputError(f"Không thể giải mã file ảnh: {source}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
            packets.append(
                FramePacket(
                    frame_id=0,
                    timestamp_sec=0.0,
                    image_rgb=rgb,
                )
            )
            return packets

    # 2. Trường hợp nguồn là file Video hoặc Webcam ID
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise InvalidInputError(f"Không thể mở nguồn video/webcam: {source}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0 and isinstance(source, str):
        cap.release()
        raise InvalidInputError(f"File video không hợp lệ hoặc rỗng: {source}")

    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 25.0  # Giá trị an toàn mặc định

    # Chế độ quét toàn video theo chuyển cảnh / chia đều (chỉ áp dụng cho file video)
    if isinstance(source, str) and sampling_mode in ("scene", "uniform"):
        scan_step = max(1, (total_frames // max(max_frames * 6, 90)) if total_frames > 0 else 1)
        candidates: List[tuple] = []
        thumbs: List[np.ndarray] = []
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break
                if frame_idx % scan_step == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
                    thumb = cv2.resize(frame_rgb, (32, 32), interpolation=cv2.INTER_AREA)
                    timestamp = frame_idx / native_fps
                    candidates.append((timestamp, frame_rgb))
                    thumbs.append(thumb)
                frame_idx += 1
        finally:
            cap.release()

        if not candidates:
            raise InvalidInputError(f"Không thể trích xuất được bất kỳ frame nào từ nguồn: {source}")

        chosen_indices = _select_scene_or_uniform_indices(thumbs, max_frames, sampling_mode)
        for out_idx, cand_idx in enumerate(chosen_indices):
            ts, img_rgb = candidates[cand_idx]
            packets.append(
                FramePacket(
                    frame_id=out_idx,
                    timestamp_sec=round(float(ts), 3),
                    image_rgb=img_rgb,
                )
            )
        return packets

    # Tính bước nhảy giữa các frame để đạt sample_fps
    step = max(1, int(round(native_fps / sample_fps)))

    frame_idx = 0
    saved_count = 0

    try:
        while cap.isOpened() and saved_count < max_frames:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            if frame_idx % step == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
                timestamp = frame_idx / native_fps
                packets.append(
                    FramePacket(
                        frame_id=saved_count,
                        timestamp_sec=round(timestamp, 3),
                        image_rgb=frame_rgb,
                    )
                )
                saved_count += 1
            frame_idx += 1
    finally:
        cap.release()

    if len(packets) == 0:
        raise InvalidInputError(f"Không đọc được frame nào từ nguồn: {source}")

    return packets
