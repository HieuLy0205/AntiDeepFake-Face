"""Dedicated M5 unit tests for src/deepfake (preprocessing, label mapping, edge cases, batching)."""

from __future__ import annotations

import numpy as np
import pytest

from src.deepfake.predictor import (
    DEFAULT_MODEL_ID,
    DeepfakeDetector,
    EnsembleDeepfakeDetector,
    LightweightCNNDetector,
    predict_deepfake,
    predict_deepfake_batch,
    resolve_fake_label_index,
)
from src.deepfake.preprocess import face_rgb_to_pil, preprocess_for_onnx, validate_rgb_face
from src.schemas import DeepfakeScore, FaceCropResult


def test_resolve_fake_label_index_variants():
    """Verify automatic label resolution across common HuggingFace id2label formats."""
    assert resolve_fake_label_index({0: "Realism", 1: "Deepfake"}) == 1
    assert resolve_fake_label_index({0: "Deepfake", 1: "Realism"}) == 0
    assert resolve_fake_label_index({0: "Fake", 1: "Real"}) == 0
    assert resolve_fake_label_index({0: "Real", 1: "Fake"}) == 1
    assert resolve_fake_label_index({"0": "authentic", "1": "ai_generated"}) == 1
    assert resolve_fake_label_index({0: "human", 1: "other"}) == 1

    with pytest.raises(ValueError):
        resolve_fake_label_index({})
    with pytest.raises(ValueError):
        resolve_fake_label_index({0: "cat", 1: "dog"})


def test_preprocess_validates_and_does_not_mutate_input(sample_face_crop_224):
    """Preprocessing must validate shape/dtype and never mutate the RGB input array."""
    original = sample_face_crop_224.copy()
    validated = validate_rgb_face(sample_face_crop_224)
    assert validated.shape == (224, 224, 3)
    assert validated.dtype == np.uint8

    pil_img = face_rgb_to_pil(sample_face_crop_224)
    assert pil_img.size == (224, 224)
    assert pil_img.mode == "RGB"

    batch = preprocess_for_onnx([sample_face_crop_224, sample_face_crop_224])
    assert batch.shape == (2, 3, 224, 224)
    assert batch.dtype == np.float32
    assert np.array_equal(sample_face_crop_224, original)

    with pytest.raises(ValueError):
        validate_rgb_face(np.zeros((224, 224), dtype=np.uint8))


def test_predict_deepfake_no_face_returns_zero_without_loading_model():
    """When has_face=False or face_rgb is None, predict_deepfake returns fake_prob=0.0 safely."""
    crop_empty = FaceCropResult(frame_id=7, has_face=False, bbox=None, det_confidence=0.0, face_rgb=None)
    score = predict_deepfake(crop_empty)
    assert isinstance(score, DeepfakeScore)
    assert score.frame_id == 7
    assert score.fake_prob == 0.0


def test_predict_deepfake_batch_mixed_valid_and_no_face(sample_face_crop_224):
    """Batched inference preserves frame_id order and skips frames without a face."""

    class _StubDetector(DeepfakeDetector):
        def __init__(self) -> None:
            self.model_id = "stub"
            self.device = "cpu"
            self.backend = "pytorch"
            self.onnx_path = None
            self.fake_label_index = 1

        def predict_probs_raw(self, faces_rgb):
            return [0.25 + 0.1 * i for i in range(len(faces_rgb))]

    stub = _StubDetector()
    crops = [
        FaceCropResult(frame_id=0, has_face=True, bbox=(0, 0, 224, 224), det_confidence=0.99, face_rgb=sample_face_crop_224),
        FaceCropResult(frame_id=1, has_face=False, bbox=None, det_confidence=0.0, face_rgb=None),
        FaceCropResult(frame_id=2, has_face=True, bbox=(0, 0, 224, 224), det_confidence=0.95, face_rgb=sample_face_crop_224),
    ]
    scores = predict_deepfake_batch(crops, detector=stub)
    assert [s.frame_id for s in scores] == [0, 1, 2]
    assert scores[0].fake_prob == pytest.approx(0.25)
    assert scores[1].fake_prob == 0.0
    assert scores[2].fake_prob == pytest.approx(0.35)


def test_ensemble_detector_weighted_average(sample_face_crop_224):
    """EnsembleDeepfakeDetector combines detector probabilities using normalized weights."""

    class _ConstDetector(DeepfakeDetector):
        def __init__(self, val: float) -> None:
            self.model_id = f"const_{val}"
            self.device = "cpu"
            self.backend = "pytorch"
            self.onnx_path = None
            self.fake_label_index = 1
            self._val = val

        def predict_probs_raw(self, faces_rgb):
            return [self._val for _ in faces_rgb]

    ens = EnsembleDeepfakeDetector(
        detectors=[_ConstDetector(0.2), _ConstDetector(0.8)],
        weights=[0.25, 0.75],
    )
    crop_ok = FaceCropResult(frame_id=3, has_face=True, bbox=(0, 0, 224, 224), det_confidence=0.99, face_rgb=sample_face_crop_224)
    crop_none = FaceCropResult(frame_id=4, has_face=False, bbox=None, det_confidence=0.0, face_rgb=None)

    single = ens.predict(crop_ok)
    assert single.frame_id == 3
    assert single.fake_prob == pytest.approx(0.65)

    batch = ens.predict_batch([crop_ok, crop_none])
    assert batch[0].fake_prob == pytest.approx(0.65)
    assert batch[1].fake_prob == 0.0
    assert DEFAULT_MODEL_ID == "king1oo1/deepfake-model"


def test_real_checkpoints_load_and_infer(sample_face_crop_224):
    """If trained M5 checkpoints exist locally, verify SigLIP2, ViT adapters and CNN load cleanly."""
    from pathlib import Path

    models_dir = Path(__file__).resolve().parents[1] / "models"
    crop = FaceCropResult(
        frame_id=0,
        has_face=True,
        bbox=(0, 0, 224, 224),
        det_confidence=0.99,
        face_rgb=sample_face_crop_224,
    )

    if (models_dir / "m5_rtfs_siglip2_adapter.pt").is_file():
        det_s = DeepfakeDetector("king1oo1/deepfake-model", device="cpu", local_files_only=True)
        assert det_s.adapter_loaded is True
        s = det_s.predict(crop)
        assert 0.0 <= s.fake_prob <= 1.0

    if (models_dir / "m5_rtfs_vit_adapter.pt").is_file():
        det_v = DeepfakeDetector("prithivMLmods/Deep-Fake-Detector-v2-Model", device="cpu", local_files_only=True)
        assert det_v.adapter_loaded is True
        v = det_v.predict(crop)
        assert 0.0 <= v.fake_prob <= 1.0

    if (models_dir / "m5_rtfs_efficientnet_b0.pt").is_file():
        det_c = LightweightCNNDetector(models_dir / "m5_rtfs_efficientnet_b0.pt", device="cpu")
        c = det_c.predict(crop)
        assert 0.0 <= c.fake_prob <= 1.0
