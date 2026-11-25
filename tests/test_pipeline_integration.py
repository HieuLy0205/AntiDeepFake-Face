"""Integration test cho run_pipeline và cầu nối UI Streamlit (M3 + M6)."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from app.components.constants import UI_ERRORS
from app.components.pipeline_bridge import (
    build_pipeline_config,
    is_backend_pipeline_available,
    run_backend_pipeline,
)
import src.pipeline as pipeline
from src import schemas

FIXTURE_FACE = Path(__file__).resolve().parent / "fixtures" / "sample_face.jpg"


def test_backend_pipeline_is_available():
    """Đảm bảo UI bridge nhận diện được `src.pipeline.run_pipeline`."""
    available, detail = is_backend_pipeline_available()
    assert available is True
    assert detail is None


def test_run_pipeline_on_sample_image(temp_valid_image_path):
    """Chạy run_pipeline đầu-cuối trên ảnh nhiễu (0 mặt) và kiểm tra cấu trúc FinalDecision."""
    config = build_pipeline_config(sample_fps=5, max_frames=10, enable_deepfake=True)
    decision = pipeline.run_pipeline(str(temp_valid_image_path), config)

    assert isinstance(decision, schemas.FinalDecision)
    assert decision.final_label in {"Real", "Printed", "On-screen", "Deepfake", "No_Face"}
    assert 0.0 <= decision.confidence <= 1.0
    assert isinstance(decision.aggregated_probs, dict)
    assert isinstance(decision.frame_logs, list)


def test_run_pipeline_on_short_video_stage1_and_stage2(temp_short_video_path):
    """Chạy run_pipeline trên video ngắn ở cả chế độ tắt và bật Deepfake (GĐ 1 & GĐ 2)."""
    video_path, _fps, _total = temp_short_video_path
    for enable_df in (False, True):
        config = build_pipeline_config(sample_fps=5, max_frames=10, enable_deepfake=enable_df)
        decision = pipeline.run_pipeline(str(video_path), config)
        assert isinstance(decision, schemas.FinalDecision)
        assert decision.final_label in {"Real", "Printed", "On-screen", "Deepfake", "No_Face"}
        assert isinstance(decision.reason, str)


def test_run_pipeline_and_bridge_on_real_face_fixture():
    """Chạy qua pipeline_bridge trên ảnh khuôn mặt thật `sample_face.jpg` và xác nhận preview + crop M3."""
    assert FIXTURE_FACE.exists(), "Thiếu fixture tests/fixtures/sample_face.jpg"
    config = build_pipeline_config(sample_fps=5, max_frames=10, enable_deepfake=True)

    res = run_backend_pipeline(str(FIXTURE_FACE), config)
    assert res.error_message is None
    assert res.is_mock_preview is False
    assert res.final_label == "Real"
    assert res.confidence > 0.5
    assert res.preview_frame_rgb is not None
    assert res.preview_bbox is not None and len(res.preview_bbox) == 4
    assert res.preview_face_crop_rgb is not None
    assert res.preview_face_crop_rgb.shape == (224, 224, 3)
    assert res.preview_face_crop_rgb.dtype == np.uint8
    assert len(res.frame_logs) == 1
    assert res.frame_logs[0]["has_face"] is True


def test_run_pipeline_on_blank_image_returns_no_face(tmp_path):
    """Ảnh trống (0 khuôn mặt) phải trả về final_label == 'No_Face'."""
    blank_path = tmp_path / "blank.png"
    cv2.imwrite(str(blank_path), np.full((240, 320, 3), 128, dtype=np.uint8))

    config = build_pipeline_config()
    decision = pipeline.run_pipeline(str(blank_path), config)
    assert decision.final_label == "No_Face"
    assert decision.confidence == 0.0


def test_run_backend_pipeline_corrupt_file_maps_ui_error(temp_corrupt_file_path):
    """Tệp hỏng qua pipeline_bridge phải map InvalidInputError về UI_ERRORS['invalid_file']."""
    config = build_pipeline_config()
    res = run_backend_pipeline(str(temp_corrupt_file_path), config)
    assert res.error_message == UI_ERRORS["invalid_file"]
