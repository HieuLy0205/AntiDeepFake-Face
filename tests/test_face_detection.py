import os
import cv2
import numpy as np
import pytest

from src.schemas import FramePacket, FaceCropResult
import src.face_detection.detector as detector_mod
from src.face_detection.detector import (
    FaceDetector,
    crop_face,
    detect_and_crop,
    get_last_face_count,
)

FIXTURE_FACE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_face.jpg")


@pytest.fixture
def detector():
    return FaceDetector()


def test_detect_no_face(detector):
    # Ảnh màu đơn sắc (không có khuôn mặt)
    blank_rgb = np.full((300, 300, 3), 128, dtype=np.uint8)
    pkt = FramePacket(frame_id=1, timestamp_sec=0.1, image_rgb=blank_rgb)

    res = detector.detect_and_crop(pkt)
    assert isinstance(res, FaceCropResult)
    assert res.frame_id == 1
    assert res.has_face is False
    assert res.bbox is None
    assert res.det_confidence == 0.0
    assert res.face_rgb is None


def test_detect_single_face(detector):
    assert os.path.exists(FIXTURE_FACE), "Thiếu file fixture sample_face.jpg"
    bgr = cv2.imread(FIXTURE_FACE)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pkt = FramePacket(frame_id=2, timestamp_sec=0.2, image_rgb=rgb)

    res = detector.detect_and_crop(pkt)
    assert isinstance(res, FaceCropResult)
    assert res.frame_id == 2
    assert res.has_face is True
    assert res.bbox is not None
    assert len(res.bbox) == 4  # (x1, y1, x2, y2)
    assert res.det_confidence > 0.5
    assert res.face_rgb is not None
    assert res.face_rgb.shape == (224, 224, 3)
    assert res.face_rgb.dtype == np.uint8


def test_detect_multiple_faces_selects_largest(detector):
    # Ghép 2 mặt: mặt 1 kích thước gốc, mặt 2 thu nhỏ 50%
    bgr = cv2.imread(FIXTURE_FACE)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape

    small_face = cv2.resize(rgb, (w // 2, h // 2))
    canvas = np.zeros((h, w + w // 2, 3), dtype=np.uint8)
    canvas[:h, :w] = rgb
    canvas[:h // 2, w:] = small_face

    pkt = FramePacket(frame_id=3, timestamp_sec=0.3, image_rgb=canvas)
    res = detector.detect_and_crop(pkt)

    assert res.has_face is True
    assert detector.last_face_count >= 2
    x1, _y1, _x2, _y2 = res.bbox
    assert x1 < w, "Hệ thống phải chọn mặt có diện tích lớn nhất (bên trái)"


def test_crop_face_multiscale_preserves_contract():
    # Kiểm tra crop đa tỉ lệ (scale_ratio lớn hơn bao phủ vùng xung quanh rộng hơn, đầu ra vẫn 224x224 uint8)
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    # Vùng mặt trung tâm màu trắng (150..250), viền ngoài màu xám (100..300)
    canvas[100:300, 100:300] = 50
    canvas[150:250, 150:250] = 255
    bbox = (150, 150, 250, 250)

    crop_tight = crop_face(canvas, bbox, target_size=(224, 224), scale_ratio=1.0)
    crop_wide = crop_face(canvas, bbox, target_size=(224, 224), scale_ratio=1.6)

    assert crop_tight is not None and crop_wide is not None
    assert crop_tight.shape == (224, 224, 3) and crop_tight.dtype == np.uint8
    assert crop_wide.shape == (224, 224, 3) and crop_wide.dtype == np.uint8
    # Crop 1.0x chỉ chứa vùng mặt trắng (mean ~ 255); crop 1.6x lấy thêm viền xám nên mean thấp hơn
    assert crop_tight.mean() > crop_wide.mean()


def test_module_level_detect_and_crop():
    bgr = cv2.imread(FIXTURE_FACE)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pkt = FramePacket(frame_id=5, timestamp_sec=0.5, image_rgb=rgb)

    res = detect_and_crop(pkt, target_size=(160, 160))
    assert res.has_face is True
    assert res.face_rgb.shape == (160, 160, 3)
    assert res.face_rgb.dtype == np.uint8


def test_detect_and_crop_single_face_mocked(monkeypatch):
    frame_rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    pkt = FramePacket(frame_id=1, timestamp_sec=0.2, image_rgb=frame_rgb)
    monkeypatch.setattr(
        detector_mod,
        "_detect_raw_boxes",
        lambda _img: [((120, 80, 320, 320), 0.98)],
    )
    res = detect_and_crop(pkt, target_size=(224, 224))
    assert res.has_face is True
    assert res.bbox == (120, 80, 320, 320)
    assert get_last_face_count() == 1
    assert res.face_rgb is not None
    assert res.face_rgb.shape == (224, 224, 3)
    assert res.face_rgb.dtype == np.uint8


def test_detect_and_crop_multiple_faces_picks_largest_bbox_mocked(monkeypatch):
    small_box = (10, 10, 60, 60)
    largest_box = (150, 100, 450, 420)
    monkeypatch.setattr(
        detector_mod,
        "_detect_raw_boxes",
        lambda _img: [(small_box, 0.99), (largest_box, 0.95)],
    )
    frame_rgb = np.full((480, 640, 3), 180, dtype=np.uint8)
    pkt = FramePacket(frame_id=2, timestamp_sec=0.4, image_rgb=frame_rgb)
    res = detect_and_crop(pkt, target_size=(224, 224))

    assert res.has_face is True
    assert res.bbox == largest_box
    assert get_last_face_count() == 2
    assert res.face_rgb is not None
    assert res.face_rgb.shape == (224, 224, 3)
    assert res.face_rgb.dtype == np.uint8


def test_detect_all_and_crop_returns_multiple_faces(detector):
    """detect_all_and_crop phải trả về nhiều khuôn mặt trong cùng khung hình, mặt lớn nhất đứng đầu."""
    bgr = cv2.imread(FIXTURE_FACE)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape

    second_face = cv2.resize(rgb, (int(w * 0.75), int(h * 0.75)))
    canvas = np.zeros((h, w + second_face.shape[1], 3), dtype=np.uint8)
    canvas[:h, :w] = rgb
    canvas[: second_face.shape[0], w:] = second_face

    pkt = FramePacket(frame_id=7, timestamp_sec=0.7, image_rgb=canvas)
    crops = detector.detect_all_and_crop(pkt, target_size=(224, 224), max_faces=4)

    assert len(crops) >= 2
    assert crops[0].bbox[0] < w  # Mặt lớn nhất bên trái đứng đầu
    assert all(c.has_face and c.face_rgb is not None and c.face_rgb.shape == (224, 224, 3) for c in crops)


def test_detect_all_and_crop_catches_crowd_including_small_background_faces(detector):
    """Nhận diện đám đông phải bắt được cả người đứng gần lẫn người đứng xa kích thước nhỏ."""
    bgr = cv2.imread(FIXTURE_FACE)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    scene = np.full((360, 640, 3), 180, dtype=np.uint8)
    # 5 người với kích thước ảnh giảm dần (từ tiền cảnh 160px xuống hậu cảnh 48px — mặt ~14px)
    sizes = [160, 110, 80, 60, 48]
    x_cursor = 12
    for sz in sizes:
        resized = cv2.resize(rgb, (sz, sz))
        scene[40 : 40 + sz, x_cursor : x_cursor + sz] = resized
        x_cursor += sz + 14

    pkt = FramePacket(frame_id=9, timestamp_sec=0.9, image_rgb=scene)
    crops = detector.detect_all_and_crop(pkt, target_size=(224, 224), max_faces=20)
    assert len(crops) == 5
    assert all(c.has_face and c.face_rgb is not None and c.face_rgb.shape == (224, 224, 3) for c in crops)


def test_nms_merges_overlapping_profile_and_ear_boxes():
    """NMS phải gộp 2 khung lệch nhẹ trên cùng 1 đầu người và geometry phải loại tai/mốc sai."""
    from src.face_detection.detector import _is_valid_face_geometry, _nms_face_boxes

    # 2 khung chồng 1 phần trên cùng đầu người quay nghiêng (mặt + tai)
    merged = _nms_face_boxes(
        [
            ((100, 50, 210, 200), 0.96),
            ((145, 45, 250, 195), 0.88),
        ],
        iou_threshold=0.25,
    )
    assert len(merged) == 1
    assert merged[0][0] == (100, 50, 210, 200)

    # Mốc mắt nằm dưới miệng (lỗi / không phải khuôn mặt) phải bị loại
    bad_landmarks = np.array(
        [[120, 160], [170, 160], [145, 130], [125, 90], [165, 90]],
        dtype=np.float32,
    )
    assert _is_valid_face_geometry((100, 50, 200, 200), bad_landmarks) is False

    # Mốc hợp lệ (mắt -> mũi -> miệng)
    good_landmarks = np.array(
        [[125, 95], [165, 95], [145, 125], [130, 160], [160, 160]],
        dtype=np.float32,
    )
    assert _is_valid_face_geometry((100, 50, 200, 200), good_landmarks) is True
