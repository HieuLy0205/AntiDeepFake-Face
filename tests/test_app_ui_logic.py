"""Unit test cho toàn bộ logic UI, validation, visualizer và webcam của M6 (app/components)."""

import io
from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.constants import (  # noqa: E402
    AGGREGATED_CLASSES,
    SPOOF_CLASSES,
    UI_ERRORS,
    VALID_FINAL_LABELS,
)
from app.components.pipeline_bridge import (  # noqa: E402
    build_pipeline_config,
    run_ui_preview_simulation,
    temporary_memory_file,
)
from app.components.validators import (  # noqa: E402
    validate_image_input,
    validate_video_input,
)
from app.components.visualizer import (  # noqa: E402
    build_frame_timeline_rows,
    collect_decision_warnings,
    compute_performance_stats,
    draw_face_annotation,
    is_near_threshold,
)
from app.components.webcam_handler import SlidingFrameBuffer  # noqa: E402


def test_frozen_label_and_error_constants():
    """Đảm bảo giữ đúng 5 chuỗi nhãn và đủ 6 thông báo lỗi UI chuẩn hóa (Mục II.6)."""
    assert VALID_FINAL_LABELS == ("Real", "Printed", "On-screen", "Deepfake", "No_Face")
    assert AGGREGATED_CLASSES == ("Real", "Printed", "On-screen", "Deepfake")
    assert SPOOF_CLASSES == ("Real", "Printed", "On-screen")
    assert UI_ERRORS["invalid_file"] == "Tệp không hợp lệ. Vui lòng chọn ảnh JPG, PNG hoặc video hợp lệ."
    assert UI_ERRORS["video_too_large_or_long"] == "Video vượt giới hạn cho phép. Vui lòng chọn video ngắn hơn."
    assert UI_ERRORS["no_face"] == "Không tìm thấy khuôn mặt. Hãy thử ảnh hoặc video rõ mặt hơn."
    assert UI_ERRORS["multi_face"] == "Phát hiện nhiều khuôn mặt. Hệ thống phân tích khuôn mặt lớn nhất."
    assert (
        UI_ERRORS["webcam_unavailable"]
        == "Không truy cập được webcam. Hãy kiểm tra quyền truy cập hoặc dùng chế độ upload."
    )
    assert UI_ERRORS["uncertain_score"] == "Kết quả chưa chắc chắn. Hãy thử thêm frame hoặc cải thiện ánh sáng."


def test_validate_image_input_valid_and_invalid():
    """Kiểm tra validate ảnh JPG/PNG hợp lệ, sai định dạng và tệp hỏng."""
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", (120, 120), color=(200, 100, 50)).save(buf, format="PNG")
    valid_bytes = buf.getvalue()

    res_ok = validate_image_input("face.png", valid_bytes)
    assert res_ok.is_valid is True
    assert res_ok.width == 120 and res_ok.height == 120

    res_bad_ext = validate_image_input("doc.pdf", valid_bytes)
    assert res_bad_ext.is_valid is False
    assert res_bad_ext.error_message == UI_ERRORS["invalid_file"]

    res_corrupt = validate_image_input("broken.jpg", b"corrupted-bytes")
    assert res_corrupt.is_valid is False
    assert res_corrupt.error_message == UI_ERRORS["invalid_file"]


def test_validate_video_input_limits():
    """Kiểm tra từ chối video sai định dạng, vượt dung lượng hoặc vượt thời lượng."""
    res_bad_ext = validate_video_input("clip.txt", size_bytes=1024)
    assert res_bad_ext.is_valid is False
    assert res_bad_ext.error_message == UI_ERRORS["invalid_file"]

    res_too_big = validate_video_input("big.mp4", size_bytes=60 * 1024 * 1024, max_size_mb=50)
    assert res_too_big.is_valid is False
    assert res_too_big.error_message == UI_ERRORS["video_too_large_or_long"]

    res_too_long = validate_video_input(
        "long.mp4",
        size_bytes=5 * 1024 * 1024,
        duration_sec_override=45.0,
        max_duration_sec=30.0,
    )
    assert res_too_long.is_valid is False
    assert res_too_long.error_message == UI_ERRORS["video_too_large_or_long"]

    res_ok = validate_video_input(
        "short.mp4",
        size_bytes=2 * 1024 * 1024,
        duration_sec_override=8.5,
    )
    assert res_ok.is_valid is True


def test_draw_face_annotation_preserves_rgb_uint8():
    """Hàm vẽ bounding box phải giữ nguyên shape (H, W, 3) và dtype uint8."""
    np = pytest.importorskip("numpy")
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    out = draw_face_annotation(img, (40, 30, 200, 210), "Real", 0.92)
    assert out.shape == (240, 320, 3)
    assert out.dtype == np.uint8
    assert out.sum() > 0


def test_near_threshold_and_multi_face_warnings():
    """Kiểm tra phát hiện điểm sát ngưỡng, nhiều khuôn mặt và không có khuôn mặt."""
    probs_near = {"Real": 0.42, "Printed": 0.30, "On-screen": 0.28, "Deepfake": 0.10}
    # spoof_score = 0.58, cách tau_spoof=0.60 đúng 0.02 <= margin=0.05
    assert is_near_threshold("Real", probs_near, tau_spoof=0.60, tau_deepfake=0.60, margin=0.05) is True

    warnings = collect_decision_warnings(
        final_label="Real",
        aggregated_probs=probs_near,
        frame_logs=[{"frame_id": 0, "face_count": 2}],
        tau_spoof=0.60,
        tau_deepfake=0.60,
        uncertain_margin=0.05,
    )
    assert UI_ERRORS["multi_face"] in warnings
    assert UI_ERRORS["uncertain_score"] in warnings

    no_face_warnings = collect_decision_warnings(
        final_label="No_Face",
        aggregated_probs={},
        frame_logs=[],
        tau_spoof=0.60,
        tau_deepfake=0.60,
        uncertain_margin=0.05,
    )
    assert no_face_warnings == [UI_ERRORS["no_face"]]


def test_timeline_and_performance_stats():
    """Kiểm tra chuẩn hóa dòng thời gian frame và tính FPS/latency."""
    logs = [
        {
            "frame_id": 0,
            "timestamp_sec": 0.0,
            "frame_label": "Real",
            "spoof_probs": {"Real": 0.9, "Printed": 0.05, "On-screen": 0.05},
            "fake_prob": 0.1,
            "latency_ms": 20.0,
        },
        {
            "frame_id": 1,
            "timestamp_sec": 0.2,
            "frame_label": "Real",
            "spoof_probs": {"Real": 0.85, "Printed": 0.1, "On-screen": 0.05},
            "fake_prob": 0.15,
            "latency_ms": 30.0,
        },
    ]
    rows = build_frame_timeline_rows(logs)
    assert len(rows) == 2
    assert rows[0]["Real"] == 0.9
    assert rows[1]["Deepfake"] == 0.15

    perf = compute_performance_stats(logs, total_elapsed_sec=0.25)
    assert perf["num_frames"] == 2.0
    assert perf["effective_fps"] == 8.0
    assert perf["avg_latency_ms"] == 25.0


def test_temporary_memory_file_deletes_after_use():
    """Đảm bảo tệp tạm xử lý xong phải bị xóa ngay (DD8: không lưu dữ liệu người dùng)."""
    captured_path = None
    with temporary_memory_file(b"transient-bytes", suffix=".jpg") as path_str:
        captured_path = Path(path_str)
        assert captured_path.exists()
    assert captured_path is not None and not captured_path.exists()


def test_sliding_frame_buffer_maxlen():
    """Bộ đệm trượt webcam chỉ giữ tối đa N frame gần nhất (DD1)."""
    np = pytest.importorskip("numpy")
    buf = SlidingFrameBuffer(max_frames=3)
    for i in range(5):
        buf.push(np.full((64, 64, 3), fill_value=i, dtype=np.uint8))
    frames = buf.get_frames()
    assert len(frames) == 3
    assert int(frames[0][0, 0, 0]) == 2
    assert int(frames[-1][0, 0, 0]) == 4


def test_build_config_and_ui_preview_simulation():
    """Kiểm tra tạo config pipeline và chế độ dry-run giao diện."""
    np = pytest.importorskip("numpy")
    cfg = build_pipeline_config(sample_fps=5, max_frames=10, tau_spoof=0.6, tau_deepfake=0.65, enable_deepfake=False)
    assert cfg["sample_fps"] == 5
    assert cfg["max_frames"] == 10
    assert cfg["enable_deepfake"] is False

    res = run_ui_preview_simulation([np.zeros((120, 120, 3), dtype=np.uint8)], cfg)
    assert res.is_mock_preview is True
    assert res.final_label in VALID_FINAL_LABELS
    assert res.preview_bbox is not None
