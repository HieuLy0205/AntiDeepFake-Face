"""Integration test cho run_pipeline từ đầu đến cuối (Mục II.7)."""

import pytest


def test_run_pipeline_on_sample_image(temp_valid_image_path):
    """Chạy run_pipeline đầu-cuối trên ảnh mẫu và kiểm tra cấu trúc FinalDecision."""
    pipeline = pytest.importorskip("src.pipeline")
    schemas = pytest.importorskip("src.schemas")

    config = {
        "sample_fps": 5,
        "max_frames": 10,
        "enable_deepfake": True,
        "thresholds": {"tau_spoof": 0.60, "tau_deepfake": 0.60, "uncertain_margin": 0.05},
    }
    decision = pipeline.run_pipeline(str(temp_valid_image_path), config)

    assert isinstance(decision, schemas.FinalDecision)
    assert decision.final_label in {"Real", "Printed", "On-screen", "Deepfake", "No_Face"}
    assert 0.0 <= decision.confidence <= 1.0
    assert isinstance(decision.aggregated_probs, dict)
    assert isinstance(decision.frame_logs, list)


def test_run_pipeline_on_short_video_stage1_and_stage2(temp_short_video_path):
    """Chạy run_pipeline trên video ngắn ở cả chế độ tắt và bật Deepfake (GĐ 1 & GĐ 2)."""
    pipeline = pytest.importorskip("src.pipeline")
    schemas = pytest.importorskip("src.schemas")

    video_path, _fps, _total = temp_short_video_path
    for enable_df in (False, True):
        config = {
            "sample_fps": 5,
            "max_frames": 10,
            "enable_deepfake": enable_df,
            "thresholds": {"tau_spoof": 0.60, "tau_deepfake": 0.60, "uncertain_margin": 0.05},
        }
        decision = pipeline.run_pipeline(str(video_path), config)
        assert isinstance(decision, schemas.FinalDecision)
        assert decision.final_label in {"Real", "Printed", "On-screen", "Deepfake", "No_Face"}
        assert isinstance(decision.reason, str)
