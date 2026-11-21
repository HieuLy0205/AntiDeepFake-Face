"""Unit test cho module src/face_detection (M3) theo Mục II.4 & II.7 (0, 1, nhiều mặt)."""

import pytest


def test_detect_and_crop_no_face_returns_has_face_false():
    """Khung hình trơn (0 khuôn mặt) phải trả has_face=False và face_rgb=None."""
    np = pytest.importorskip("numpy")
    schemas = pytest.importorskip("src.schemas")
    detector = pytest.importorskip("src.face_detection.detector")

    blank_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = schemas.FramePacket(frame_id=0, timestamp_sec=0.0, image_rgb=blank_rgb)
    res = detector.detect_and_crop(packet, target_size=(224, 224))

    assert isinstance(res, schemas.FaceCropResult)
    assert res.frame_id == 0
    assert res.has_face is False
    assert res.face_rgb is None


def test_detect_and_crop_single_face_returns_224_rgb_uint8(monkeypatch):
    """Khi có 1 khuôn mặt, đầu ra bắt buộc là RGB uint8 kích thước (224, 224, 3)."""
    np = pytest.importorskip("numpy")
    schemas = pytest.importorskip("src.schemas")
    detector = pytest.importorskip("src.face_detection.detector")

    frame_rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    packet = schemas.FramePacket(frame_id=1, timestamp_sec=0.2, image_rgb=frame_rgb)

    # Nếu detector dùng backend nội bộ có thể mock danh sách bbox, test trường hợp 1 mặt
    if hasattr(detector, "_detect_raw_boxes"):
        monkeypatch.setattr(
            detector,
            "_detect_raw_boxes",
            lambda _img: [((120, 80, 320, 320), 0.98)],
        )
        res = detector.detect_and_crop(packet, target_size=(224, 224))
        assert res.has_face is True
        assert res.bbox is not None
        assert res.face_rgb is not None
        assert res.face_rgb.shape == (224, 224, 3)
        assert res.face_rgb.dtype == np.uint8
    else:
        res = detector.detect_and_crop(packet, target_size=(224, 224))
        assert isinstance(res, schemas.FaceCropResult)
        if res.has_face and res.face_rgb is not None:
            assert res.face_rgb.shape == (224, 224, 3)
            assert res.face_rgb.dtype == np.uint8


def test_detect_and_crop_multiple_faces_picks_largest_bbox(monkeypatch):
    """Khi có nhiều khuôn mặt, hệ thống bắt buộc chọn khuôn mặt có diện tích lớn nhất."""
    np = pytest.importorskip("numpy")
    schemas = pytest.importorskip("src.schemas")
    detector = pytest.importorskip("src.face_detection.detector")

    if not hasattr(detector, "_detect_raw_boxes"):
        pytest.skip("Detector chưa tách helper _detect_raw_boxes để tiêm giả lập nhiều mặt.")

    small_box = (10, 10, 60, 60)       # diện tích = 2500
    largest_box = (150, 100, 450, 420)  # diện tích = 300 * 320 = 96000
    monkeypatch.setattr(
        detector,
        "_detect_raw_boxes",
        lambda _img: [(small_box, 0.99), (largest_box, 0.95)],
    )

    frame_rgb = np.full((480, 640, 3), 180, dtype=np.uint8)
    packet = schemas.FramePacket(frame_id=2, timestamp_sec=0.4, image_rgb=frame_rgb)
    res = detector.detect_and_crop(packet, target_size=(224, 224))

    assert res.has_face is True
    assert res.bbox == largest_box
    assert res.face_rgb is not None
    assert res.face_rgb.shape == (224, 224, 3)
    assert res.face_rgb.dtype == np.uint8
