import os
import tempfile
import cv2
import numpy as np
import pytest

from src.schemas import FramePacket
from src.video_io.reader import (
    InvalidInputError,
    SlidingFrameBuffer,
    load_frames,
    load_single_frame,
)


def test_load_frames_non_existent_file():
    with pytest.raises(InvalidInputError):
        load_frames("non_existent_file_path_12345.mp4")


def test_load_frames_corrupted_file():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"corrupted video content that opencv cannot open")
        temp_path = f.name
    try:
        with pytest.raises(InvalidInputError):
            load_frames(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_load_frames_invalid_sampling_params(temp_valid_image_path):
    with pytest.raises(InvalidInputError):
        load_frames(str(temp_valid_image_path), sample_fps=0, max_frames=10)
    with pytest.raises(InvalidInputError):
        load_frames(str(temp_valid_image_path), sample_fps=5, max_frames=0)


def test_load_frames_static_image():
    # Tạo ảnh có góc trên bên trái màu Đỏ thuần túy theo RGB: [255, 0, 0]
    # Lưu bằng cv2 (BGR) thì phải lưu là [0, 0, 255]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_bgr[:, :] = [0, 0, 255]  # Đỏ trong BGR
        cv2.imwrite(f.name, img_bgr)
        temp_path = f.name

    try:
        packets = load_frames(temp_path)
        assert len(packets) == 1
        pkt = packets[0]
        assert isinstance(pkt, FramePacket)
        assert pkt.frame_id == 0
        assert pkt.timestamp_sec == 0.0
        assert pkt.image_rgb.shape == (100, 100, 3)
        assert pkt.image_rgb.dtype == np.uint8
        # Kiểm tra màu đã chuyển đúng sang RGB: pixel phải là [255, 0, 0]
        assert np.array_equal(pkt.image_rgb[0, 0], [255, 0, 0]), "Lỗi: Ảnh chưa được chuyển sang hệ màu RGB!"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_load_frames_synthetic_video():
    # Tạo video ngắn: 30 FPS, độ dài 30 frames (1 giây)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_path = f.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, 30.0, (120, 120))
    for i in range(30):
        frame = np.full((120, 120, 3), i * 8, dtype=np.uint8)
        out.write(frame)
    out.release()

    try:
        # Lấy mẫu với sample_fps = 5, tối đa 10 frames
        # Với 30 FPS và sample_fps = 5, step = 30 / 5 = 6 frames -> 5 frames
        packets = load_frames(temp_path, sample_fps=5, max_frames=10)
        assert len(packets) == 5
        assert packets[0].frame_id == 0
        assert packets[1].frame_id == 1
        assert packets[0].image_rgb.shape == (120, 120, 3)
        assert packets[0].image_rgb.dtype == np.uint8
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_load_single_frame():
    img_rgb = np.full((200, 200, 3), 128, dtype=np.uint8)
    pkt = load_single_frame(img_rgb, frame_id=99)
    assert isinstance(pkt, FramePacket)
    assert pkt.frame_id == 99
    assert pkt.image_rgb.shape == (200, 200, 3)
    assert pkt.image_rgb.dtype == np.uint8


def test_sliding_frame_buffer():
    buf = SlidingFrameBuffer(max_frames=3)
    for i in range(5):
        pkt = FramePacket(
            frame_id=i,
            timestamp_sec=float(i) * 0.2,
            image_rgb=np.full((64, 64, 3), i, dtype=np.uint8),
        )
        buf.push(pkt)

    assert len(buf) == 3
    frames = buf.get_frames()
    assert [f.frame_id for f in frames] == [2, 3, 4]
    buf.clear()
    assert len(buf) == 0

    with pytest.raises(InvalidInputError):
        SlidingFrameBuffer(max_frames=0)


def test_load_frames_scene_and_uniform_span_full_video():
    """Chế độ scene và uniform phải trích xuất khung hình trải dài toàn bộ thời lượng video."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_path = f.name

    # Video 60 frame @ 30 FPS (2.0 giây) với 3 cảnh khác màu rõ rệt ở 0s, 0.7s, 1.4s
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, 30.0, (120, 120))
    for i in range(60):
        if i < 20:
            val = (20, 20, 20)
        elif i < 40:
            val = (220, 40, 40)
        else:
            val = (40, 220, 220)
        frame = np.full((120, 120, 3), val, dtype=np.uint8)
        out.write(frame)
    out.release()

    try:
        scene_pkts = load_frames(temp_path, max_frames=5, sampling_mode="scene")
        assert len(scene_pkts) == 5
        assert scene_pkts[0].timestamp_sec == 0.0
        assert scene_pkts[-1].timestamp_sec >= 1.3
        assert [p.frame_id for p in scene_pkts] == [0, 1, 2, 3, 4]

        uniform_pkts = load_frames(temp_path, max_frames=4, sampling_mode="uniform")
        assert len(uniform_pkts) == 4
        assert uniform_pkts[0].timestamp_sec == 0.0
        assert uniform_pkts[-1].timestamp_sec >= 1.5
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
