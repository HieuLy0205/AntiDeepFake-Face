import pytest
import numpy as np

def test_frame_packet_creation():
    from src.schemas import FramePacket
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = FramePacket(frame_id=0, timestamp_sec=0.0, image_rgb=img)
    assert packet.frame_id == 0
    assert packet.timestamp_sec == 0.0
    assert packet.image_rgb.shape == (480, 640, 3)
    assert packet.image_rgb.dtype == np.uint8

def test_face_crop_result_no_face():
    from src.schemas import FaceCropResult
    res = FaceCropResult(frame_id=1, has_face=False)
    assert res.frame_id == 1
    assert res.has_face is False
    assert res.bbox is None
    assert res.det_confidence == 0.0
    assert res.face_rgb is None

def test_face_crop_result_with_face():
    from src.schemas import FaceCropResult
    face_img = np.zeros((224, 224, 3), dtype=np.uint8)
    res = FaceCropResult(
        frame_id=2,
        has_face=True,
        bbox=(50, 60, 150, 180),
        det_confidence=0.95,
        face_rgb=face_img
    )
    assert res.has_face is True
    assert res.bbox == (50, 60, 150, 180)
    assert res.det_confidence == 0.95
    assert res.face_rgb.shape == (224, 224, 3)
    assert res.face_rgb.dtype == np.uint8
