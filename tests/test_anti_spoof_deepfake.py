"""Kiểm thử hợp đồng đầu ra cho src/anti_spoof (M4) và src/deepfake (M5)."""

import math
import pytest


def test_predict_spoof_returns_three_normalized_probabilities(sample_face_crop_224):
    """SpoofScore.probs phải có đủ 3 key chuẩn và tổng xác suất = 1.0."""
    schemas = pytest.importorskip("src.schemas")
    anti_spoof = pytest.importorskip("src.anti_spoof.predictor")

    crop = schemas.FaceCropResult(
        frame_id=0,
        has_face=True,
        bbox=(10, 10, 234, 234),
        det_confidence=0.99,
        face_rgb=sample_face_crop_224,
    )
    score = anti_spoof.predict_spoof(crop)

    assert isinstance(score, schemas.SpoofScore)
    assert score.frame_id == 0
    assert set(score.probs.keys()) == {"Real", "Printed", "On-screen"}
    for prob in score.probs.values():
        assert 0.0 <= prob <= 1.0
    assert math.isclose(sum(score.probs.values()), 1.0, rel_tol=1e-3, abs_tol=1e-3)


def test_predict_deepfake_returns_valid_probability(sample_face_crop_224):
    """DeepfakeScore.fake_prob phải nằm trong [0.0, 1.0]."""
    schemas = pytest.importorskip("src.schemas")
    deepfake = pytest.importorskip("src.deepfake.predictor")

    crop = schemas.FaceCropResult(
        frame_id=0,
        has_face=True,
        bbox=(10, 10, 234, 234),
        det_confidence=0.99,
        face_rgb=sample_face_crop_224,
    )
    score = deepfake.predict_deepfake(crop)

    assert isinstance(score, schemas.DeepfakeScore)
    assert score.frame_id == 0
    assert 0.0 <= score.fake_prob <= 1.0
