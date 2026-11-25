import os
import tempfile
import cv2
import numpy as np
import pytest

from src.schemas import FramePacket, FaceCropResult
from src.video_io.reader import InvalidInputError, load_frames
from src.face_detection.detector import FaceDetector

FIXTURE_FACE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_face.jpg")


@pytest.fixture
def detector():
    return FaceDetector()


# 1. Ca biên: File hỏng và không tồn tại
def test_edge_case_non_existent_and_corrupt_files():
    with pytest.raises(InvalidInputError):
        load_frames("this_file_does_not_exist_xyz.mp4")

    # File dung lượng 0 byte
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        zero_byte_path = f.name
    try:
        with pytest.raises(InvalidInputError):
            load_frames(zero_byte_path)
    finally:
        if os.path.exists(zero_byte_path):
            os.remove(zero_byte_path)


# 2. Ca biên: 0 khuôn mặt (ảnh đen hoàn toàn, ảnh trắng hoàn toàn, ảnh xám)
@pytest.mark.parametrize("color", [0, 255, 128])
def test_edge_case_zero_face(detector, color):
    img = np.full((320, 320, 3), color, dtype=np.uint8)
    pkt = FramePacket(frame_id=0, timestamp_sec=0.0, image_rgb=img)
    result = detector.detect_and_crop(pkt)

    assert result.has_face is False
    assert result.bbox is None
    assert result.det_confidence == 0.0
    assert result.face_rgb is None


# 3. Ca biên: Khuôn mặt nằm sát mép ảnh (Border Clamping)
def test_edge_case_face_at_boundary(detector):
    assert os.path.exists(FIXTURE_FACE)
    bgr = cv2.imread(FIXTURE_FACE)
    face_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fh, fw, _ = face_rgb.shape

    # Đặt mặt ở góc trên cùng bên trái (chạm viền x=0, y=0)
    canvas = np.zeros((fh + 100, fw + 100, 3), dtype=np.uint8)
    canvas[:fh, :fw] = face_rgb

    pkt = FramePacket(frame_id=1, timestamp_sec=0.0, image_rgb=canvas)

    # Scale ratio lớn (1.8x) sẽ khiến vùng crop vượt ra ngoài biên âm nhưng vẫn phải clamp an toàn
    result = detector.detect_and_crop(pkt, scale_ratio=1.8)
    assert result.has_face is True
    assert result.bbox[0] >= 0, "Tọa độ x1 không được nhỏ hơn 0"
    assert result.bbox[1] >= 0, "Tọa độ y1 không được nhỏ hơn 0"
    assert result.face_rgb is not None
    assert result.face_rgb.shape == (224, 224, 3)
    assert result.face_rgb.dtype == np.uint8


# 4. Ca biên: Nhiều khuôn mặt và kích thước cực lớn / nhỏ
def test_edge_case_multiple_faces_varying_sizes(detector):
    assert os.path.exists(FIXTURE_FACE)
    bgr = cv2.imread(FIXTURE_FACE)
    face_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Tạo ảnh ghép 3 mặt: lớn (100%), vừa (60%), nhỏ (30%)
    h, w, _ = face_rgb.shape
    med_face = cv2.resize(face_rgb, (int(w * 0.6), int(h * 0.6)))
    small_face = cv2.resize(face_rgb, (int(w * 0.3), int(h * 0.3)))

    canvas = np.zeros((h, w + int(w * 0.6) + int(w * 0.3), 3), dtype=np.uint8)
    canvas[:h, :w] = face_rgb
    canvas[:med_face.shape[0], w:w + med_face.shape[1]] = med_face
    canvas[:small_face.shape[0], w + med_face.shape[1]:] = small_face

    pkt = FramePacket(frame_id=2, timestamp_sec=0.0, image_rgb=canvas)
    result = detector.detect_and_crop(pkt)

    assert result.has_face is True
    # Đảm bảo chọn mặt lớn nhất (bên trái nhất, x1 < w)
    assert result.bbox[0] < w
    assert result.face_rgb.shape == (224, 224, 3)
    assert result.face_rgb.dtype == np.uint8


# 5. Ca biên: Đảm bảo tính bất biến của kiểu dữ liệu đầu ra RGB uint8 [0, 255]
def test_edge_case_output_contract_invariance(detector):
    bgr = cv2.imread(FIXTURE_FACE)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pkt = FramePacket(frame_id=10, timestamp_sec=1.5, image_rgb=rgb)

    result = detector.detect_and_crop(pkt)
    assert result.has_face is True
    assert result.face_rgb.dtype == np.uint8, "Bắt buộc dtype phải là uint8"
    assert result.face_rgb.shape == (224, 224, 3), "Bắt buộc shape phải là (224, 224, 3)"
    assert result.face_rgb.min() >= 0
    assert result.face_rgb.max() <= 255
