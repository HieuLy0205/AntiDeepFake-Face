"""Unit test cho toàn bộ logic UI, validation, visualizer và webcam của M6 (app/components)."""

import io
import json
from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.constants import (  # noqa: E402
    ACADEMIC_LIMITATION_BANNER,
    AGGREGATED_CLASSES,
    SPOOF_CLASSES,
    UI_ERRORS,
    VALID_FINAL_LABELS,
)
from app.components.pipeline_bridge import (  # noqa: E402
    build_pipeline_config,
    run_backend_frames_pipeline,
    run_ui_preview_simulation,
    temporary_memory_file,
)
from app.components.validators import (  # noqa: E402
    validate_image_input,
    validate_video_input,
)
from app.components.visualizer import (  # noqa: E402
    build_export_report_dict,
    build_frame_timeline_rows,
    collect_decision_warnings,
    compute_performance_stats,
    draw_face_annotation,
    draw_multi_face_annotations,
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
    assert res.preview_face_crop_rgb is not None
    assert res.preview_face_crop_rgb.shape == (224, 224, 3)


def test_build_export_report_dict_is_json_serializable():
    """Báo cáo xuất JSON phải thuần dữ liệu (không chứa ndarray) và giữ đủ khóa chuẩn."""
    np = pytest.importorskip("numpy")
    cfg = build_pipeline_config(sample_fps=5, max_frames=5, enable_deepfake=True)
    res = run_ui_preview_simulation([np.zeros((160, 160, 3), dtype=np.uint8)], cfg)
    report = build_export_report_dict(res, cfg, input_mode="Upload ảnh", source_name="demo.jpg")

    encoded = json.dumps(report, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["final_label"] == res.final_label
    assert decoded["input_mode"] == "Upload ảnh"
    assert decoded["source_name"] == "demo.jpg"
    assert len(decoded["frame_timeline"]) == 1
    assert set(decoded["aggregated_probs"].keys()) == set(AGGREGATED_CLASSES)


def test_run_backend_frames_pipeline_multi_frame(monkeypatch):
    """Kiểm tra chạy pipeline đa frame từ bộ đệm webcam trình duyệt qua pipeline_bridge."""
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    import src.pipeline as pipeline
    from src import schemas

    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_face.jpg"
    if fixture_path.is_file():
        with Image.open(fixture_path) as img:
            frame = np.asarray(img.convert("RGB"), dtype=np.uint8)
    else:
        frame = np.full((240, 240, 3), 120, dtype=np.uint8)

    original_loader = pipeline._load_optional_callable

    def stubbed_loader(module_path, attr_name):
        if module_path == "src.deepfake.predictor" and attr_name == "predict_deepfake":
            return lambda crop: schemas.DeepfakeScore(frame_id=crop.frame_id, fake_prob=0.05)
        if module_path == "src.anti_spoof.predictor" and attr_name == "predict_spoof":
            return lambda crop: schemas.SpoofScore(
                frame_id=crop.frame_id,
                probs={"Real": 0.92, "Printed": 0.04, "On-screen": 0.04},
                latency_ms=10.0,
            )
        return original_loader(module_path, attr_name)

    monkeypatch.setattr(pipeline, "_load_optional_callable", stubbed_loader)
    cfg = build_pipeline_config(sample_fps=5, max_frames=3, enable_deepfake=True)
    res = run_backend_frames_pipeline([frame, frame], cfg)

    assert res.error_message is None
    assert res.is_mock_preview is False
    assert res.final_label in VALID_FINAL_LABELS
    assert len(res.frame_logs) == 2
    assert [row["frame_id"] for row in res.frame_logs] == [0, 1]


def test_streamlit_app_headless_smoke():
    """Kiểm tra khởi tạo giao diện Streamlit và tương tác cơ bản bằng AppTest."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=20)
    at.run()
    assert len(at.exception) == 0

    rendered_markdown = "\n".join(el.value for el in at.markdown)
    assert ACADEMIC_LIMITATION_BANNER in rendered_markdown

    # Kiểm tra các nút điều khiển ở sidebar & vùng chính có mặt đầy đủ
    assert any(b.label == "✨ Dùng ảnh mẫu thử nghiệm" for b in at.button)

    # Thử nhấn nút nạp ảnh mẫu
    demo_btn = next(b for b in at.button if b.label == "✨ Dùng ảnh mẫu thử nghiệm")
    demo_btn.click().run()
    assert len(at.exception) == 0
    assert any(b.label == "🔍 Phân tích khuôn mặt" for b in at.button)


def test_multi_face_annotations_and_scene_previews(monkeypatch):
    """Kiểm tra vẽ HUD đa khuôn mặt và trả về frame_previews từ pipeline_bridge."""
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    import src.pipeline as pipeline
    from src import schemas
    from app.components.pipeline_bridge import run_backend_pipeline

    img = np.zeros((240, 320, 3), dtype=np.uint8)
    faces = [
        {"face_index": 1, "bbox": (20, 30, 120, 160), "label": "Deepfake", "confidence": 0.91},
        {"face_index": 2, "bbox": (180, 40, 280, 160), "label": "Real", "confidence": 0.88},
    ]
    annotated = draw_multi_face_annotations(img, faces)
    assert annotated.shape == (240, 320, 3)
    assert annotated.dtype == np.uint8
    assert annotated.sum() > 0

    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_face.jpg"
    if fixture_path.is_file():
        with Image.open(fixture_path) as f_img:
            rgb = np.asarray(f_img.convert("RGB"), dtype=np.uint8)
        h, w, _ = rgb.shape
        second = np.asarray(f_img.convert("RGB").resize((int(w * 0.7), int(h * 0.7))), dtype=np.uint8)
        canvas = np.zeros((h, w + second.shape[1], 3), dtype=np.uint8)
        canvas[:h, :w] = rgb
        canvas[: second.shape[0], w:] = second

        original_loader = pipeline._load_optional_callable

        def stubbed_loader(module_path, attr_name):
            if module_path == "src.deepfake.predictor" and attr_name == "predict_deepfake":
                return lambda crop: schemas.DeepfakeScore(frame_id=crop.frame_id, fake_prob=0.10)
            if module_path == "src.anti_spoof.predictor" and attr_name == "predict_spoof":
                return lambda crop: schemas.SpoofScore(
                    frame_id=crop.frame_id,
                    probs={"Real": 0.90, "Printed": 0.05, "On-screen": 0.05},
                    latency_ms=8.0,
                )
            return original_loader(module_path, attr_name)

        monkeypatch.setattr(pipeline, "_load_optional_callable", stubbed_loader)
        cfg = build_pipeline_config(
            sample_fps=5,
            max_frames=5,
            enable_deepfake=True,
            sampling_mode="scene",
            multi_face=True,
            include_frame_previews=True,
        )
        with temporary_memory_file(
            fixture_path.read_bytes(), suffix=".jpg"
        ) as _:
            pass

        buf = io.BytesIO()
        Image.fromarray(canvas, mode="RGB").save(buf, format="JPEG")
        with temporary_memory_file(buf.getvalue(), suffix=".jpg") as tmp_img:
            res = run_backend_pipeline(tmp_img, cfg)
        assert res.error_message is None
        assert len(res.frame_previews) == 1
        assert len(res.frame_previews[0]["faces"]) >= 2
        # Đảm bảo JSON export vẫn serializable dù có frame_previews
        report = build_export_report_dict(res, cfg, input_mode="Upload ảnh", source_name="multi.jpg")
        assert len(json.dumps(report, ensure_ascii=False)) > 0


def test_pipeline_dual_scale_and_highest_risk_strategy(monkeypatch):
    """GĐ 1 phải nhận crop 2.7x rộng hơn GĐ 2 (1.2x), và highest_risk phải bắt được người phụ Deepfake."""
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    import src.pipeline as pipeline
    from src import schemas
    from app.components.pipeline_bridge import run_backend_pipeline

    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_face.jpg"
    if not fixture_path.is_file():
        pytest.skip("Thiếu sample_face.jpg")

    with Image.open(fixture_path) as f_img:
        rgb = np.asarray(f_img.convert("RGB"), dtype=np.uint8)
    h, w, _ = rgb.shape
    second = np.asarray(f_img.convert("RGB").resize((int(w * 0.65), int(h * 0.65))), dtype=np.uint8)
    canvas = np.zeros((h, w + second.shape[1], 3), dtype=np.uint8)
    canvas[:h, :w] = rgb
    canvas[: second.shape[0], w:] = second

    spoof_means = []
    df_means = []
    original_loader = pipeline._load_optional_callable

    def stubbed_loader(module_path, attr_name):
        if module_path == "src.anti_spoof.predictor" and attr_name == "predict_spoof":
            def _spoof(crop):
                spoof_means.append(float(crop.face_rgb.mean()))
                return schemas.SpoofScore(
                    frame_id=crop.frame_id,
                    probs={"Real": 0.94, "Printed": 0.03, "On-screen": 0.03},
                )
            return _spoof
        if module_path == "src.deepfake.predictor" and attr_name == "predict_deepfake":
            def _df(crop):
                df_means.append(float(crop.face_rgb.mean()))
                # Người chính bên trái (bbox x1 < w) là Real (0.05); người phụ bên phải là Deepfake (0.93)
                is_second = crop.bbox is not None and crop.bbox[0] >= w
                return schemas.DeepfakeScore(frame_id=crop.frame_id, fake_prob=0.93 if is_second else 0.05)
            return _df
        return original_loader(module_path, attr_name)

    monkeypatch.setattr(pipeline, "_load_optional_callable", stubbed_loader)

    buf = io.BytesIO()
    Image.fromarray(canvas, mode="RGB").save(buf, format="JPEG")
    with temporary_memory_file(buf.getvalue(), suffix=".jpg") as tmp_img:
        cfg_primary = build_pipeline_config(
            multi_face=True,
            include_frame_previews=True,
            aggregation_strategy="primary",
        )
        res_primary = run_backend_pipeline(tmp_img, cfg_primary)
        assert res_primary.final_label == "Real"

        # Kiểm tra crop GĐ 1 (2.7x) khác crop GĐ 2 (1.2x) do bao phủ nền đen xung quanh
        assert spoof_means and df_means and spoof_means[0] != pytest.approx(df_means[0], abs=0.5)

        cfg_risk = build_pipeline_config(
            multi_face=True,
            include_frame_previews=True,
            aggregation_strategy="highest_risk",
        )
        res_risk = run_backend_pipeline(tmp_img, cfg_risk)
        assert res_risk.final_label == "Deepfake"
