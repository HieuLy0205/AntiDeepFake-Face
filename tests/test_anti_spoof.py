import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.anti_spoof.model import AntiSpoofModel
from src.schemas import FaceCropResult, SpoofScore


def test_predict_combines_normalized_level_probabilities(sample_face_crop_224):
    """Level 1 thật,giả và Level 2 in,màn hình phải ghép đúng xác suất."""
    model = AntiSpoofModel(
        level1_predictor=lambda crop: (0.2, 0.8),
        level2_predictor=lambda crop: (0.3, 0.7),
    )

    probabilities = model.predict(sample_face_crop_224)

    assert set(probabilities) == {"real", "printed", "onscreen"}
    assert probabilities == pytest.approx(
        {"real": 0.2, "printed": 0.24, "onscreen": 0.56}
    )
    assert math.isclose(sum(probabilities.values()), 1.0)


def test_predict_skips_level2_below_spoof_threshold(sample_face_crop_224):
    """Không chạy phân loại in/màn hình khi điểm giả chưa đạt ngưỡng."""
    level2_calls = []

    def level2_predictor(crop):
        level2_calls.append(crop)
        return 0.1, 0.9

    model = AntiSpoofModel(
        spoof_threshold=0.5,
        level1_predictor=lambda crop: (0.8, 0.2),
        level2_predictor=level2_predictor,
    )

    probabilities = model.predict(sample_face_crop_224)

    assert level2_calls == []
    assert probabilities == pytest.approx(
        {"real": 0.8, "printed": 0.1, "onscreen": 0.1}
    )


def test_default_level1_uses_predict_level1_from_module(
    sample_face_crop_224, monkeypatch
):
    """Mặc định phải gọi predictor Level 1 thật, không dùng kết quả mock cũ."""
    received_crops = []

    def fake_predict_level1(crop):
        received_crops.append(crop)
        return 0.9, 0.1

    monkeypatch.setitem(
        sys.modules,
        "src.anti_spoof.level1",
        SimpleNamespace(predict_level1=fake_predict_level1),
    )
    model = AntiSpoofModel(
        level2_predictor=lambda crop: pytest.fail("Level 2 should not be called")
    )

    probabilities = model.predict(sample_face_crop_224)

    assert len(received_crops) == 1
    assert received_crops[0] is sample_face_crop_224
    assert probabilities == pytest.approx(
        {"real": 0.9, "printed": 0.05, "onscreen": 0.05}
    )


def test_default_level2_uses_predict_level2_from_module(
    sample_face_crop_224, monkeypatch
):
    """Khi Level 1 phát hiện spoof, mặc định gọi heuristic Level 2."""
    received_crops = []

    def fake_predict_level2(crop):
        received_crops.append(crop)
        return 0.85, 0.15

    monkeypatch.setitem(
        sys.modules,
        "src.anti_spoof.level2",
        SimpleNamespace(predict_level2=fake_predict_level2),
    )
    model = AntiSpoofModel(level1_predictor=lambda crop: (0.05, 0.95))

    probabilities = model.predict(sample_face_crop_224)

    assert len(received_crops) == 1
    assert received_crops[0] is sample_face_crop_224
    assert probabilities == pytest.approx(
        {"real": 0.05, "printed": 0.8075, "onscreen": 0.1425}
    )
    assert math.isclose(sum(probabilities.values()), 1.0)


@pytest.mark.parametrize(
    "crop",
    [
        None,
        np.empty((0, 224, 3), dtype=np.uint8),
        np.zeros((224, 224), dtype=np.uint8),
        np.zeros((224, 224, 4), dtype=np.uint8),
        np.zeros((224, 224, 3), dtype=np.float32),
    ],
)
def test_predict_rejects_invalid_crop(crop):
    """Crop sai kích thước, số kênh hoặc dtype phải báo lỗi rõ ràng."""
    with pytest.raises(ValueError, match="RGB uint8"):
        AntiSpoofModel().predict(crop)


def test_predict_spoof_adapts_to_pipeline_contract(
    sample_face_crop_224, monkeypatch
):
    """Adapter đổi output M4 sang SpoofScore mà pipeline hiện đang sử dụng."""
    from src.anti_spoof import predictor

    monkeypatch.setattr(
        predictor,
        "_model",
        AntiSpoofModel(
            level1_predictor=lambda crop: (0.2, 0.8),
            level2_predictor=lambda crop: (0.3, 0.7),
        ),
    )
    crop = FaceCropResult(
        frame_id=7,
        has_face=True,
        bbox=(10, 10, 234, 234),
        det_confidence=0.99,
        face_rgb=sample_face_crop_224,
    )

    score = predictor.predict_spoof(crop)

    assert isinstance(score, SpoofScore)
    assert score.frame_id == 7
    assert set(score.probs) == {"Real", "Printed", "On-screen"}
    assert score.probs == pytest.approx(
        {"Real": 0.2, "Printed": 0.24, "On-screen": 0.56}
    )
    assert math.isclose(sum(score.probs.values()), 1.0)
    assert score.latency_ms >= 0.0


def test_predict_spoof_rejects_crop_without_face():
    """Không gửi crop không có khuôn mặt xuống model."""
    from src.anti_spoof.predictor import predict_spoof

    with pytest.raises(ValueError, match="detected face"):
        predict_spoof(FaceCropResult(frame_id=0, has_face=False))
