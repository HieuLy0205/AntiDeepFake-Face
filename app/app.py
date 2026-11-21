<<<<<<< HEAD

=======
"""Ứng dụng Web Streamlit phát hiện gian lận khuôn mặt — AntiDeepFake-Face (M6)."""

import io
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import streamlit as st

# Đảm bảo import được `app.*` và `src.*` dù chạy streamlit từ thư mục nào
_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from app.components.constants import (  # noqa: E402
    ACADEMIC_LIMITATION_BANNER,
    AGGREGATED_CLASSES,
    DD6_PRECEDENCE_NOTE,
    DEFAULT_MAX_FRAMES,
    DEFAULT_N_FRAMES,
    DEFAULT_SAMPLE_FPS,
    DEFAULT_TAU_DEEPFAKE,
    DEFAULT_TAU_SPOOF,
    DEFAULT_UNCERTAIN_MARGIN,
    LABEL_DEEPFAKE,
    LABEL_DISPLAY_VI,
    LABEL_NO_FACE,
    LABEL_ONSCREEN,
    LABEL_PRINTED,
    LABEL_REAL,
    MAX_VIDEO_DURATION_SEC,
    MAX_VIDEO_SIZE_MB,
    UI_ERRORS,
)
from app.components.pipeline_bridge import (  # noqa: E402
    PipelineExecutionResult,
    build_pipeline_config,
    is_backend_pipeline_available,
    run_backend_pipeline,
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
    normalize_aggregated_probs,
)
from app.components.webcam_handler import (  # noqa: E402
    SlidingFrameBuffer,
    capture_webcam_burst,
    check_webcam_available,
)


def _load_css() -> None:
    css_path = Path(__file__).resolve().parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Đang khởi tạo tài nguyên pipeline...")
def get_cached_backend_status() -> Dict[str, Any]:
    """Nạp và cache trạng thái pipeline/model một lần theo DD8."""
    available, detail = is_backend_pipeline_available()
    return {"available": available, "detail": detail}


def _decode_rgb_image(raw_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(raw_bytes)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _verdict_css_modifier(label: str) -> str:
    mapping = {
        LABEL_REAL: "verdict-card--real",
        LABEL_PRINTED: "verdict-card--printed",
        LABEL_ONSCREEN: "verdict-card--onscreen",
        LABEL_DEEPFAKE: "verdict-card--deepfake",
        LABEL_NO_FACE: "verdict-card--noface",
    }
    return mapping.get(label, "verdict-card--noface")


def _render_limitation_banner() -> None:
    st.markdown(
        f"""
        <div class="limitation-banner">
            <strong>Thông báo giới hạn hệ thống:</strong> {ACADEMIC_LIMITATION_BANNER}<br/>
            <span>{DD6_PRECEDENCE_NOTE}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_results(
    result: PipelineExecutionResult,
    tau_spoof: float,
    tau_deepfake: float,
    uncertain_margin: float,
    enable_deepfake: bool,
) -> None:
    if result.error_message:
        st.error(result.error_message)
        return

    if result.is_mock_preview:
        st.warning(
            "Đang chạy ở chế độ xem trước UI (chưa kết nối `src/pipeline.py`). "
            "Số liệu bên dưới chỉ dùng kiểm tra giao diện."
        )

    probs = normalize_aggregated_probs(result.aggregated_probs)
    warnings = collect_decision_warnings(
        final_label=result.final_label,
        aggregated_probs=probs,
        frame_logs=result.frame_logs,
        tau_spoof=tau_spoof,
        tau_deepfake=tau_deepfake,
        uncertain_margin=uncertain_margin,
    )

    for warn in warnings:
        if result.final_label == LABEL_NO_FACE:
            st.error(warn)
        else:
            st.warning(warn)

    col_left, col_right = st.columns([1.05, 1.15], gap="large")

    with col_left:
        st.subheader("Khung hình phân tích")
        if result.preview_frame_rgb is not None:
            annotated = draw_face_annotation(
                frame_rgb=result.preview_frame_rgb,
                bbox=result.preview_bbox,
                label=result.final_label,
                confidence=result.confidence,
            )
            st.image(annotated, caption="Khung hình đại diện kèm bounding box", use_container_width=True)
        else:
            st.info("Không có khung hình đại diện để hiển thị.")

    with col_right:
        st.subheader("Kết luận tổng hợp")
        display_label = LABEL_DISPLAY_VI.get(result.final_label, result.final_label)
        card_mod = _verdict_css_modifier(result.final_label)
        st.markdown(
            f"""
            <div class="verdict-card {card_mod}">
                <div class="verdict-title">{display_label}</div>
                <p class="verdict-meta">
                    <strong>Độ tin cậy:</strong> {result.confidence * 100:.1f}%<br/>
                    <strong>Lý do quyết định:</strong> {result.reason or "Không có"}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Xác suất từng nhánh (trung bình qua các frame)**")
        for cls_name in AGGREGATED_CLASSES:
            if cls_name == LABEL_DEEPFAKE and not enable_deepfake:
                st.caption("Deepfake (GĐ 2): Đã tắt theo cấu hình.")
                continue
            val = probs.get(cls_name, 0.0)
            c1, c2 = st.columns([3.5, 1])
            with c1:
                st.progress(val, text=f"{cls_name}")
            with c2:
                st.markdown(f"`{val * 100:.1f}%`")

        perf = compute_performance_stats(result.frame_logs, result.elapsed_sec)
        m1, m2, m3 = st.columns(3)
        m1.metric("Số frame xử lý", int(perf["num_frames"]))
        m2.metric("Tốc độ (FPS)", f"{perf['effective_fps']:.2f}")
        m3.metric("Độ trễ TB (ms)", f"{perf['avg_latency_ms']:.1f}")

    st.divider()
    st.subheader("Dòng thời gian nhãn theo frame")
    timeline_rows = build_frame_timeline_rows(result.frame_logs)
    if timeline_rows:
        chart_series: Dict[str, List[float]] = {
            LABEL_REAL: [float(r["Real"]) for r in timeline_rows],
            LABEL_PRINTED: [float(r["Printed"]) for r in timeline_rows],
            LABEL_ONSCREEN: [float(r["On-screen"]) for r in timeline_rows],
        }
        if enable_deepfake:
            chart_series[LABEL_DEEPFAKE] = [
                float(r["Deepfake"]) if r["Deepfake"] is not None else 0.0 for r in timeline_rows
            ]
        st.line_chart(chart_series)
        with st.expander("Xem log chi tiết từng frame", expanded=False):
            st.dataframe(timeline_rows, use_container_width=True)
    else:
        st.caption("Chưa có log từng frame được trả về từ pipeline.")


def _execute_file_analysis(
    raw_bytes: bytes,
    suffix: str,
    config: Dict[str, Any],
    fallback_frame_rgb: Optional[np.ndarray],
    backend_available: bool,
    allow_ui_preview: bool,
    is_video: bool,
) -> Optional[PipelineExecutionResult]:
    if not backend_available and not allow_ui_preview:
        st.error(
            "Pipeline `src/pipeline.py` chưa sẵn sàng để chạy suy luận thật. "
            "Vui lòng tích hợp module `src/` hoặc bật 'Chế độ xem trước UI' ở thanh bên để kiểm tra giao diện."
        )
        return None

    progress_bar = st.progress(0, text="Đang chuẩn bị dữ liệu trong bộ nhớ phiên...") if is_video else None
    try:
        if not backend_available and allow_ui_preview:
            if progress_bar:
                progress_bar.progress(60, text="Đang chạy mô phỏng giao diện...")
            frames = [fallback_frame_rgb] if fallback_frame_rgb is not None else [
                np.zeros((240, 320, 3), dtype=np.uint8)
            ]
            res = run_ui_preview_simulation(frames, config)
            if progress_bar:
                progress_bar.progress(100, text="Hoàn tất kiểm tra giao diện.")
            return res

        with temporary_memory_file(raw_bytes, suffix=suffix) as temp_path:
            if progress_bar:
                progress_bar.progress(30, text="Đang lấy frame và chạy các nhánh mô hình...")
            res = run_backend_pipeline(temp_path, config, fallback_frame_rgb=fallback_frame_rgb)
            if progress_bar:
                progress_bar.progress(100, text="Phân tích hoàn tất.")
            return res
    finally:
        if progress_bar:
            progress_bar.empty()


def main() -> None:
    st.set_page_config(
        page_title="AntiDeepFake-Face | Phát hiện gian lận khuôn mặt",
        page_icon="🛡️",
        layout="wide",
    )
    _load_css()

    st.title("AntiDeepFake-Face — Hệ thống Phát hiện Gian lận Khuôn mặt")
    _render_limitation_banner()

    backend_status = get_cached_backend_status()
    backend_available = bool(backend_status["available"])

    with st.sidebar:
        st.header("Tùy chỉnh kiểm tra")
        input_mode = st.radio(
            "Nguồn đầu vào",
            options=["Upload ảnh", "Upload video", "Webcam"],
            index=0,
        )

        st.subheader("Lấy mẫu khung hình")
        st.caption("Áp dụng cho video hoặc webcam để tổng hợp nhiều khung hình ổn định hơn.")
        preset_n = st.selectbox(
            "Mức khung hình nhanh",
            options=["Tùy chỉnh", 1, 5, 10, 30],
            index=0,
        )
        n_frames = st.slider(
            "Số khung hình phân tích",
            min_value=1,
            max_value=DEFAULT_MAX_FRAMES,
            value=DEFAULT_N_FRAMES if preset_n == "Tùy chỉnh" else int(preset_n),
            step=1,
            help="Số lượng khung hình tối đa được lấy để tính kết quả trung bình.",
        )
        sample_fps = st.slider(
            "Số khung hình mỗi giây",
            min_value=1,
            max_value=15,
            value=DEFAULT_SAMPLE_FPS,
            step=1,
            help="Tốc độ trích xuất khung hình từ video đầu vào.",
        )

        st.subheader("Ngưỡng quyết định")
        st.caption("Tăng ngưỡng để giảm cảnh báo nhầm; giảm ngưỡng để kiểm tra khắt khe hơn.")
        tau_spoof = st.slider(
            "Ngưỡng phát hiện giả mạo (ảnh in / màn hình)",
            min_value=0.10,
            max_value=0.95,
            value=DEFAULT_TAU_SPOOF,
            step=0.05,
        )
        enable_deepfake = st.toggle(
            "Kiểm tra khuôn mặt Deepfake",
            value=True,
            help="Bật để phân tích thêm dấu hiệu khuôn mặt bị hoán đổi hoặc tạo bằng AI.",
        )
        tau_deepfake = st.slider(
            "Ngưỡng phát hiện Deepfake",
            min_value=0.10,
            max_value=0.95,
            value=DEFAULT_TAU_DEEPFAKE,
            step=0.05,
            disabled=not enable_deepfake,
        )
        uncertain_margin = st.slider(
            "Khoảng cảnh báo chưa chắc chắn",
            min_value=0.00,
            max_value=0.20,
            value=DEFAULT_UNCERTAIN_MARGIN,
            step=0.01,
            help="Hiển thị nhắc nhở khi điểm số nằm sát ngưỡng quyết định.",
        )

        allow_ui_preview = False
        if not backend_available:
            st.info("Chế độ xem trước giao diện (chưa kết nối mô hình nhận diện).")
            allow_ui_preview = st.checkbox("Cho phép chạy thử giao diện", value=True)

    config = build_pipeline_config(
        sample_fps=sample_fps,
        max_frames=n_frames,
        tau_spoof=tau_spoof,
        tau_deepfake=tau_deepfake,
        uncertain_margin=uncertain_margin,
        enable_deepfake=enable_deepfake,
    )

    result: Optional[PipelineExecutionResult] = None

    if input_mode == "Upload ảnh":
        uploaded = st.file_uploader(
            "Chọn ảnh khuôn mặt (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            val = validate_image_input(uploaded.name, raw_bytes)
            if not val.is_valid:
                st.error(val.error_message or UI_ERRORS["invalid_file"])
            else:
                rgb_img = _decode_rgb_image(raw_bytes)
                st.image(rgb_img, caption=f"Ảnh đã chọn ({val.width}x{val.height})", width=360)
                if st.button("Phân tích", type="primary", use_container_width=False):
                    suffix = Path(uploaded.name).suffix.lower() or ".jpg"
                    with st.spinner("Đang phân tích khuôn mặt..."):
                        result = _execute_file_analysis(
                            raw_bytes=raw_bytes,
                            suffix=suffix,
                            config=config,
                            fallback_frame_rgb=rgb_img,
                            backend_available=backend_available,
                            allow_ui_preview=allow_ui_preview,
                            is_video=False,
                        )

    elif input_mode == "Upload video":
        st.caption(
            f"Giới hạn khuyến nghị: tối đa {MAX_VIDEO_SIZE_MB} MB và {MAX_VIDEO_DURATION_SEC} giây."
        )
        uploaded = st.file_uploader(
            "Chọn video ngắn (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            suffix = Path(uploaded.name).suffix.lower() or ".mp4"
            with temporary_memory_file(raw_bytes, suffix=suffix) as probe_path:
                val = validate_video_input(
                    filename=uploaded.name,
                    size_bytes=len(raw_bytes),
                    video_path=probe_path,
                )
            if not val.is_valid:
                st.error(val.error_message or UI_ERRORS["invalid_file"])
            else:
                st.success(
                    f"Video hợp lệ: {val.size_mb:.2f} MB"
                    + (f", thời lượng ~{val.duration_sec:.1f}s" if val.duration_sec is not None else "")
                )
                if st.button("Phân tích", type="primary"):
                    result = _execute_file_analysis(
                        raw_bytes=raw_bytes,
                        suffix=suffix,
                        config=config,
                        fallback_frame_rgb=None,
                        backend_available=backend_available,
                        allow_ui_preview=allow_ui_preview,
                        is_video=True,
                    )

    else:
        st.subheader("Chế độ Webcam")
        webcam_mode = st.radio(
            "Phương thức đọc webcam",
            options=["Camera trình duyệt (st.camera_input)", "Webcam cục bộ (OpenCV ID 0)"],
            horizontal=True,
        )

        if "webcam_buffer" not in st.session_state:
            st.session_state["webcam_buffer"] = SlidingFrameBuffer(max_frames=DEFAULT_MAX_FRAMES)
        buffer: SlidingFrameBuffer = st.session_state["webcam_buffer"]

        if webcam_mode == "Camera trình duyệt (st.camera_input)":
            cam_photo = st.camera_input("Chụp ảnh từ webcam (hỗ trợ chụp nhiều lần để gộp N frame)")
            if cam_photo is not None:
                raw_bytes = cam_photo.getvalue()
                rgb_cam = _decode_rgb_image(raw_bytes)
                buffer.push(rgb_cam)
                st.caption(f"Số frame hiện có trong bộ đệm trượt: {len(buffer)} / {DEFAULT_MAX_FRAMES}")

                c_run, c_clear = st.columns([1, 1])
                run_clicked = c_run.button("Phân tích", type="primary")
                if c_clear.button("Xóa bộ đệm webcam"):
                    buffer.clear()
                    st.rerun()

                if run_clicked:
                    with st.spinner("Đang phân tích ảnh webcam..."):
                        result = _execute_file_analysis(
                            raw_bytes=raw_bytes,
                            suffix=".jpg",
                            config=config,
                            fallback_frame_rgb=rgb_cam,
                            backend_available=backend_available,
                            allow_ui_preview=allow_ui_preview,
                            is_video=False,
                        )
        else:
            device_id = int(st.number_input("Webcam Device ID", min_value=0, max_value=5, value=0, step=1))
            if st.button("Phân tích", type="primary"):
                ok_cam, cam_err = check_webcam_available(device_id=device_id)
                if not ok_cam:
                    st.error(cam_err or UI_ERRORS["webcam_unavailable"])
                elif backend_available:
                    with st.spinner("Đang đọc webcam và chạy pipeline..."):
                        result = run_backend_pipeline(device_id, config)
                elif allow_ui_preview:
                    burst = capture_webcam_burst(device_id=device_id, num_frames=min(n_frames, 5))
                    if not burst.success:
                        st.error(burst.error_message or UI_ERRORS["webcam_unavailable"])
                    else:
                        result = run_ui_preview_simulation(burst.frames_rgb, config)
                else:
                    st.error(UI_ERRORS["webcam_unavailable"])

    if result is not None:
        st.divider()
        _render_results(
            result=result,
            tau_spoof=tau_spoof,
            tau_deepfake=tau_deepfake,
            uncertain_margin=uncertain_margin,
            enable_deepfake=enable_deepfake,
        )


if __name__ == "__main__":
    main()
>>>>>>> M6
