"""Ứng dụng Web Streamlit phát hiện gian lận khuôn mặt — AntiDeepFake-Face (M6)."""

from datetime import datetime
import hashlib
import html
import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import altair as alt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Đảm bảo import được package `app.*` và `src.*` khi chạy `streamlit run app/app.py`
# (Streamlit tự thêm thư mục `app/` vào đầu sys.path khiến `app/app.py` che mất package `app`)
_APP_DIR = str(Path(__file__).resolve().parent)
_ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path = [_ROOT_DIR] + [p for p in sys.path if p not in (_ROOT_DIR, _APP_DIR)]
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

# Tự động nạp lại các module `app.*` / `src.*` nếu server Streamlit đang giữ bản cũ trong RAM
_EXPECTED_CODE_REV = "2026.11.v5"
_loaded_bridge = sys.modules.get("app.components.pipeline_bridge")
if _loaded_bridge is not None and getattr(_loaded_bridge, "_ADF_CODE_REV", None) != _EXPECTED_CODE_REV:
    for _mod_name in list(sys.modules):
        if _mod_name == "app" or _mod_name.startswith(("app.components", "src")):
            del sys.modules[_mod_name]

from app.components.constants import (  # noqa: E402
    ACADEMIC_LIMITATION_BANNER,
    AGGREGATED_CLASSES,
    CONFIG_PRESETS,
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
    LABEL_UI_META,
    MAX_IMAGE_SIZE_MB,
    MAX_VIDEO_DURATION_SEC,
    MAX_VIDEO_SIZE_MB,
    UI_ERRORS,
)
from app.components.pipeline_bridge import (  # noqa: E402
    PipelineExecutionResult,
    build_pipeline_config,
    is_backend_pipeline_available,
    query_deepfake_model_status,
    run_backend_frames_pipeline,
    run_backend_pipeline,
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
    normalize_aggregated_probs,
)
from app.components.webcam_handler import (  # noqa: E402
    SlidingFrameBuffer,
    capture_webcam_burst,
    check_webcam_available,
)

_INPUT_MODES = ["Upload ảnh", "Upload video", "Webcam"]
_SAMPLE_FACE_PATH = Path(_ROOT_DIR) / "tests" / "fixtures" / "sample_face.jpg"


def _render_html(markup: str) -> None:
    """Render HTML an toàn qua st.markdown (loại bỏ thụt đầu dòng & gắn notranslate)."""
    compact = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(f'<div class="notranslate" translate="no">{compact}</div>', unsafe_allow_html=True)


def _install_dom_translation_guard() -> None:
    """Ngăn trình duyệt dịch tự động phá cấu trúc DOM của React (lỗi removeChild)."""
    guard_html = """
    <script>
    (function() {
        function protectWindow(win) {
            if (!win || win.__adfDomGuardInstalled) return;
            win.__adfDomGuardInstalled = true;
            try {
                var doc = win.document;
                if (doc && doc.documentElement) {
                    doc.documentElement.setAttribute("translate", "no");
                    doc.documentElement.classList.add("notranslate");
                    if (!doc.documentElement.getAttribute("lang")) {
                        doc.documentElement.setAttribute("lang", "vi");
                    }
                    if (doc.head && !doc.querySelector('meta[name="google"][content="notranslate"]')) {
                        var meta = doc.createElement("meta");
                        meta.name = "google";
                        meta.content = "notranslate";
                        doc.head.appendChild(meta);
                    }
                }
                if (win.Node && win.Node.prototype) {
                    var origRemoveChild = win.Node.prototype.removeChild;
                    win.Node.prototype.removeChild = function(child) {
                        if (!child || child.parentNode !== this) {
                            return child;
                        }
                        return origRemoveChild.apply(this, arguments);
                    };
                    var origInsertBefore = win.Node.prototype.insertBefore;
                    win.Node.prototype.insertBefore = function(newNode, referenceNode) {
                        if (referenceNode && referenceNode.parentNode !== this) {
                            return origInsertBefore.call(this, newNode, null);
                        }
                        return origInsertBefore.apply(this, arguments);
                    };
                }
            } catch (e) {}
        }
        protectWindow(window);
        try {
            if (window.parent && window.parent !== window) {
                protectWindow(window.parent);
            }
        } catch (e) {}
    })();
    </script>
    """
    st.html(guard_html, unsafe_allow_javascript=True)


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


def _init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "active_mode": "Upload ảnh",
        "sidebar_mode_ctrl": "Upload ảnh",
        "main_mode_ctrl": "Upload ảnh",
        "preset_profile": "Cân bằng (Mặc định)",
        "preset_n": "Tùy chỉnh",
        "n_frames": DEFAULT_N_FRAMES,
        "sample_fps": DEFAULT_SAMPLE_FPS,
        "sampling_mode": "scene",
        "multi_face": True,
        "max_faces_per_frame": 20,
        "det_conf_threshold": 0.62,
        "aggregation_strategy": "highest_risk",
        "deepfake_variant": "siglip2",
        "live_webcam_active": False,
        "scene_filter": "Tất cả",
        "scene_grid_cols": "3 cột",
        "tau_spoof": DEFAULT_TAU_SPOOF,
        "enable_deepfake": True,
        "tau_deepfake": DEFAULT_TAU_DEEPFAKE,
        "uncertain_margin": DEFAULT_UNCERTAIN_MARGIN,
        "last_cam_digest": None,
        "demo_image_bytes": None,
        "demo_image_name": None,
        "latest_result": None,
        "latest_meta": None,
        "analysis_history": [],
        "webcam_probe_status": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if "webcam_buffer" not in st.session_state:
        st.session_state["webcam_buffer"] = SlidingFrameBuffer(max_frames=DEFAULT_MAX_FRAMES)


def _sync_mode_from_sidebar() -> None:
    chosen = st.session_state.get("sidebar_mode_ctrl")
    if chosen in _INPUT_MODES:
        st.session_state["active_mode"] = chosen
        st.session_state["main_mode_ctrl"] = chosen


def _sync_mode_from_main() -> None:
    chosen = st.session_state.get("main_mode_ctrl")
    if chosen in _INPUT_MODES:
        st.session_state["active_mode"] = chosen
        st.session_state["sidebar_mode_ctrl"] = chosen


def _on_preset_n_change() -> None:
    val = st.session_state.get("preset_n")
    if isinstance(val, int):
        st.session_state["n_frames"] = val


def _apply_preset_profile(profile_name: Optional[str] = None) -> None:
    name = profile_name or st.session_state.get("preset_profile", "Cân bằng (Mặc định)")
    cfg = CONFIG_PRESETS.get(name, CONFIG_PRESETS["Cân bằng (Mặc định)"])
    st.session_state["preset_n"] = "Tùy chỉnh"
    st.session_state["n_frames"] = int(cfg["n_frames"])
    st.session_state["sample_fps"] = int(cfg["sample_fps"])
    st.session_state["tau_spoof"] = float(cfg["tau_spoof"])
    st.session_state["enable_deepfake"] = bool(cfg["enable_deepfake"])
    st.session_state["tau_deepfake"] = float(cfg["tau_deepfake"])
    st.session_state["uncertain_margin"] = float(cfg["uncertain_margin"])
    st.session_state["det_conf_threshold"] = 0.62


def _record_result(
    result: PipelineExecutionResult,
    input_mode: str,
    source_name: str,
    config: Dict[str, Any],
) -> None:
    st.session_state["latest_result"] = result
    st.session_state["latest_meta"] = {
        "input_mode": input_mode,
        "source_name": source_name,
        "config": config,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    if result.error_message:
        return

    perf = compute_performance_stats(result.frame_logs, result.elapsed_sec)
    probs = normalize_aggregated_probs(result.aggregated_probs)
    history: List[dict] = list(st.session_state.get("analysis_history", []))
    history.insert(
        0,
        {
            "Thời điểm": st.session_state["latest_meta"]["timestamp"],
            "Chế độ": input_mode,
            "Nguồn": source_name,
            "Kết luận": result.final_label,
            "Độ tin cậy": f"{result.confidence * 100:.1f}%",
            "Spoof (%)": round((probs[LABEL_PRINTED] + probs[LABEL_ONSCREEN]) * 100.0, 1),
            "Deepfake (%)": round(probs[LABEL_DEEPFAKE] * 100.0, 1)
            if config.get("enable_deepfake", True)
            else None,
            "Số frame": int(perf["num_frames"]),
            "FPS": perf["effective_fps"],
        },
    )
    st.session_state["analysis_history"] = history[:20]


def _render_hero_and_stepper(
    backend_available: bool,
    enable_deepfake: bool,
    latest_result: Optional[PipelineExecutionResult],
) -> None:
    backend_pill = (
        '<span class="status-pill status-pill--ok">● Pipeline Sẵn sàng</span>'
        if backend_available
        else '<span class="status-pill status-pill--warn">● Chế độ Xem trước UI</span>'
    )
    stage_pill = (
        '<span class="status-pill status-pill--cyan">GĐ 1 + GĐ 2 Đang bật</span>'
        if enable_deepfake
        else '<span class="status-pill status-pill--muted">Chỉ GĐ 1 (Anti-Spoof)</span>'
    )
    privacy_pill = '<span class="status-pill status-pill--muted">🔒 RAM-Only (DD8)</span>'

    _render_html(
        f"""
        <div class="adf-hero">
            <div class="adf-hero-left">
                <div class="adf-shield-emblem">🛡️</div>
                <div>
                    <h1 class="adf-hero-title">AntiDeepFake-Face Studio</h1>
                    <p class="adf-hero-subtitle">
                        Hệ thống phát hiện giả mạo khuôn mặt đa tầng: Chống giả mạo vật lý (Ảnh in / Màn hình) &amp; Deepfake AI
                    </p>
                </div>
            </div>
            <div class="adf-hero-pills">
                {backend_pill}
                {stage_pill}
                {privacy_pill}
            </div>
        </div>
        """
    )

    _render_html(
        f"""
        <div class="limitation-banner">
            <strong>Thông báo giới hạn hệ thống:</strong> {html.escape(ACADEMIC_LIMITATION_BANNER)}<br/>
            <span>{html.escape(DD6_PRECEDENCE_NOTE)}</span>
        </div>
        """
    )

    # Xác định trạng thái 4 tầng theo kết quả hiện tại
    s1_cls = "pipeline-step pipeline-step--active"
    s2_cls = "pipeline-step pipeline-step--active"
    s3_cls = "pipeline-step pipeline-step--active" if enable_deepfake else "pipeline-step pipeline-step--disabled"
    s4_cls = "pipeline-step pipeline-step--active"

    if latest_result is not None and not latest_result.error_message:
        lbl = latest_result.final_label
        if lbl == LABEL_NO_FACE:
            s1_cls = "pipeline-step pipeline-step--triggered"
        elif lbl in (LABEL_PRINTED, LABEL_ONSCREEN):
            s1_cls = "pipeline-step pipeline-step--pass"
            s2_cls = "pipeline-step pipeline-step--triggered"
            s4_cls = "pipeline-step pipeline-step--triggered"
        elif lbl == LABEL_DEEPFAKE:
            s1_cls = "pipeline-step pipeline-step--pass"
            s2_cls = "pipeline-step pipeline-step--pass"
            if enable_deepfake:
                s3_cls = "pipeline-step pipeline-step--triggered"
            s4_cls = "pipeline-step pipeline-step--triggered"
        elif lbl == LABEL_REAL:
            s1_cls = "pipeline-step pipeline-step--pass"
            s2_cls = "pipeline-step pipeline-step--pass"
            if enable_deepfake:
                s3_cls = "pipeline-step pipeline-step--pass"
            s4_cls = "pipeline-step pipeline-step--pass"

    s3_variant_txt = {
        "siglip2": "SigLIP2",
        "ensemble": "SigLIP2 + ViT",
        "vit": "ViT v2",
    }.get(str(st.session_state.get("deepfake_variant", "siglip2")), "SigLIP2")
    s3_badge = f"Đang bật ({s3_variant_txt})" if enable_deepfake else "Đã tắt"
    _render_html(
        f"""
        <div class="pipeline-stepper">
            <div class="{s1_cls}">
                <div class="pipeline-step-head"><span>Bước 1 • M3</span><span>YuNet</span></div>
                <p class="pipeline-step-title">Phát hiện &amp; Cắt mặt</p>
                <p class="pipeline-step-desc">Cắt kép 2.7× (M4) &amp; 1.2× (M5)</p>
            </div>
            <div class="{s2_cls}">
                <div class="pipeline-step-head"><span>Bước 2 • GĐ 1 (M4)</span><span>MiniFASNet</span></div>
                <p class="pipeline-step-title">Chống giả mạo vật lý</p>
                <p class="pipeline-step-desc">Real · Printed · On-screen</p>
            </div>
            <div class="{s3_cls}">
                <div class="pipeline-step-head"><span>Bước 3 • GĐ 2 (M5)</span><span>{html.escape(s3_badge)}</span></div>
                <p class="pipeline-step-title">Phát hiện Deepfake AI</p>
                <p class="pipeline-step-desc">Batch Face Manipulation</p>
            </div>
            <div class="{s4_cls}">
                <div class="pipeline-step-head"><span>Bước 4 • M1</span><span>DD6 / DD7</span></div>
                <p class="pipeline-step-title">Tổng hợp Đa khung hình</p>
                <p class="pipeline-step-desc">Ưu tiên Spoof trước Deepfake</p>
            </div>
        </div>
        """
    )


def _render_sidebar(backend_available: bool) -> bool:
    with st.sidebar:
        _render_html(
            """
            <div class="sidebar-brand">
                <p class="sidebar-brand-title">🛡️ Bảng Điều Khiển</p>
                <p class="sidebar-brand-sub">Cấu hình lấy mẫu &amp; ngưỡng quyết định DD6</p>
            </div>
            """
        )

        _render_html('<div class="sidebar-section-label">1. Cấu hình nhanh</div>')
        st.selectbox(
            "Hồ sơ tham số",
            options=list(CONFIG_PRESETS.keys()),
            key="preset_profile",
            label_visibility="collapsed",
        )
        col_apply, col_reset = st.columns(2)
        col_apply.button(
            "Áp dụng",
            on_click=_apply_preset_profile,
            width="stretch",
            help="Áp dụng nhanh bộ tham số đã chọn.",
        )
        col_reset.button(
            "Mặc định",
            on_click=_apply_preset_profile,
            args=("Cân bằng (Mặc định)",),
            width="stretch",
            help="Khôi phục toàn bộ thông số về mặc định.",
        )

        _render_html('<div class="sidebar-section-label">2. Nguồn đầu vào</div>')
        st.radio(
            "Chọn chế độ kiểm tra",
            options=_INPUT_MODES,
            key="sidebar_mode_ctrl",
            on_change=_sync_mode_from_sidebar,
            label_visibility="collapsed",
        )

        _render_html('<div class="sidebar-section-label">3. Lấy mẫu khung hình (DD1)</div>')
        _sampling_labels = {
            "scene": "🎯 Tự động bắt từng khung cảnh (Scene)",
            "uniform": "📏 Chia đều toàn thời lượng video",
            "fps": "⏱️ Theo FPS cố định từ đầu video",
        }
        st.selectbox(
            "Chế độ tách khung video",
            options=["scene", "uniform", "fps"],
            format_func=lambda m: _sampling_labels.get(m, m),
            key="sampling_mode",
            help="Scene: ưu tiên các khung hình có chuyển cảnh khác biệt trên toàn video.",
        )
        st.toggle(
            "Nhận diện mọi người trong khung hình",
            key="multi_face",
            help="Đóng khung và chấm điểm riêng từng khuôn mặt (#1, #2, ...) xuất hiện trong mỗi cảnh.",
        )
        if st.session_state.get("multi_face"):
            _agg_labels = {
                "highest_risk": "⚠️ Ưu tiên người có rủi ro cao nhất",
                "primary": "👤 Chỉ tính theo người chính (#1)",
            }
            st.selectbox(
                "Quy tắc kết luận nhiều người",
                options=["highest_risk", "primary"],
                format_func=lambda s: _agg_labels.get(s, s),
                key="aggregation_strategy",
                help="Ưu tiên người rủi ro cao nhất giúp thẻ kết luận tổng bắt được nếu có người phụ bị giả mạo.",
            )
            st.slider(
                "Số người tối đa mỗi khung hình",
                min_value=2,
                max_value=25,
                step=1,
                key="max_faces_per_frame",
                help="Tăng lên để bắt cả những người đứng xa hoặc xung quanh trong ảnh/video đông người.",
            )
            st.slider(
                "Độ chặt lọc khuôn mặt (chống bắt nhầm tai/nền)",
                min_value=0.45,
                max_value=0.85,
                step=0.02,
                key="det_conf_threshold",
                help="Mức chuẩn 0.62–0.70 lọc sạch tai/vai mờ. Giảm xuống 0.50 nếu muốn cố bắt mặt rất nhỏ ở xa.",
            )
        st.selectbox(
            "Mức khung hình nhanh",
            options=["Tùy chỉnh", 1, 5, 10, 30],
            key="preset_n",
            on_change=_on_preset_n_change,
        )
        st.slider(
            "Số khung hình phân tích (N)",
            min_value=1,
            max_value=DEFAULT_MAX_FRAMES,
            step=1,
            key="n_frames",
            help="Số lượng khung hình tối đa được lấy từ video/webcam để tính trung bình.",
        )
        st.slider(
            "Tần suất trích xuất (FPS)",
            min_value=1,
            max_value=15,
            step=1,
            key="sample_fps",
            disabled=(st.session_state.get("sampling_mode") != "fps"),
            help="Áp dụng khi chọn chế độ tách khung theo FPS cố định hoặc webcam.",
        )

        _render_html('<div class="sidebar-section-label">4. Ngưỡng quyết định (DD6)</div>')
        st.slider(
            "Ngưỡng giả mạo vật lý (τ_spoof)",
            min_value=0.10,
            max_value=0.95,
            step=0.05,
            key="tau_spoof",
            help="Tổng xác suất (Printed + On-screen) vượt ngưỡng này sẽ kết luận Spoof.",
        )
        st.toggle(
            "Kiểm tra khuôn mặt Deepfake (GĐ 2)",
            key="enable_deepfake",
            help="Bật để chạy thêm nhánh phân tích khuôn mặt hoán đổi/tổng hợp bằng AI.",
        )
        _variant_labels = {
            "siglip2": "⚡ SigLIP2 đơn (Nhanh, có sẵn cache)",
            "ensemble": "🛡️ Ensemble SigLIP2 + ViT (Giảm báo nhầm)",
            "vit": "🔬 ViT Deepfake v2 (Đối chiếu độc lập)",
        }
        selected_variant = st.selectbox(
            "Mô hình Deepfake (GĐ 2)",
            options=["siglip2", "ensemble", "vit"],
            format_func=lambda v: _variant_labels.get(v, v),
            key="deepfake_variant",
            disabled=not bool(st.session_state["enable_deepfake"]),
        )
        m_status = query_deepfake_model_status(str(selected_variant))
        if not st.session_state.get("enable_deepfake"):
            note_html = "Nhánh kiểm tra Deepfake (GĐ 2) đang tắt."
        elif selected_variant == "siglip2" and not m_status.get("has_siglip_adapter"):
            note_html = (
                "ℹ️ Đang dùng trọng số gốc HuggingFace (chưa có <code>m5_rtfs_siglip2_adapter.pt</code>). "
                "Chọn <strong>Ensemble</strong> nếu muốn giảm báo nhầm trên video nén."
            )
        elif selected_variant in ("ensemble", "vit") and not m_status.get("vit_cached"):
            note_html = (
                "ℹ️ Model ViT v2 sẽ tự tải một lần vào <code>models/hf_cache/</code> ở lần chạy đầu."
            )
        else:
            note_html = "✅ Model Deepfake đã sẵn sàng trong cache cục bộ."
        _render_html(f'<div class="sidebar-model-note">{note_html}</div>')

        st.slider(
            "Ngưỡng phát hiện Deepfake (τ_deepfake)",
            min_value=0.10,
            max_value=0.95,
            step=0.05,
            key="tau_deepfake",
            disabled=not bool(st.session_state["enable_deepfake"]),
        )
        st.slider(
            "Khoảng cảnh báo sát ngưỡng",
            min_value=0.00,
            max_value=0.20,
            step=0.01,
            key="uncertain_margin",
            help="Hiển thị cảnh báo chưa chắc chắn khi điểm số nằm trong vùng ±margin quanh ngưỡng.",
        )

        allow_ui_preview = False
        if not backend_available:
            st.divider()
            st.info("Chưa kết nối đầy đủ model `src/pipeline.py`.")
            allow_ui_preview = st.checkbox("Cho phép chạy thử giao diện (UI Preview)", value=True)

        st.divider()
        hist_count = len(st.session_state.get("analysis_history", []))
        st.caption(f"Phiên hiện tại: **{hist_count}** lượt phân tích (lưu tạm trong RAM).")
        if st.session_state.get("latest_result") is not None:
            if st.button("Xóa kết quả hiện tại", width="stretch"):
                st.session_state["latest_result"] = None
                st.session_state["latest_meta"] = None
                st.rerun()

    return allow_ui_preview


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
            frames = (
                [fallback_frame_rgb]
                if fallback_frame_rgb is not None
                else [np.zeros((240, 320, 3), dtype=np.uint8)]
            )
            res = run_ui_preview_simulation(frames, config)
            if progress_bar:
                progress_bar.progress(100, text="Hoàn tất kiểm tra giao diện.")
            return res

        with temporary_memory_file(raw_bytes, suffix=suffix) as temp_path:
            if progress_bar:
                progress_bar.progress(35, text="Đang trích xuất khung hình và chạy các nhánh mô hình...")
            res = run_backend_pipeline(temp_path, config, fallback_frame_rgb=fallback_frame_rgb)
            if progress_bar:
                progress_bar.progress(100, text="Phân tích hoàn tất.")
            return res
    finally:
        if progress_bar:
            progress_bar.empty()


def _build_confidence_ring_svg(confidence: float, accent_hex: str) -> str:
    pct = max(0.0, min(100.0, confidence * 100.0))
    radius = 34
    circumference = 2.0 * 3.14159265 * radius
    offset = circumference * (1.0 - pct / 100.0)
    return (
        f'<svg width="88" height="88" viewBox="0 0 88 88" aria-label="Độ tin cậy">'
        f'<circle cx="44" cy="44" r="{radius}" fill="none" stroke="rgba(148,163,184,0.2)" stroke-width="7"/>'
        f'<circle cx="44" cy="44" r="{radius}" fill="none" stroke="{accent_hex}" stroke-width="7" '
        f'stroke-linecap="round" stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}" '
        f'transform="rotate(-90 44 44)"/>'
        f'<text x="44" y="48" text-anchor="middle" fill="#f8fafc" font-size="15" font-weight="700" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{pct:.1f}%</text>'
        f'</svg>'
    )


def _render_prob_bars_html(
    probs: Dict[str, float],
    tau_spoof: float,
    tau_deepfake: float,
    enable_deepfake: bool,
) -> str:
    spoof_total = max(0.0, min(1.0, probs.get(LABEL_PRINTED, 0.0) + probs.get(LABEL_ONSCREEN, 0.0)))
    stage1_real = probs.get(LABEL_REAL, 0.0)
    deepfake_p = probs.get(LABEL_DEEPFAKE, 0.0)
    effective_real = (
        max(0.0, min(stage1_real, 1.0 - deepfake_p)) if enable_deepfake else stage1_real
    )
    bars: List[Dict[str, Any]] = [
        {
            "name": "Xác suất Người thật thực tế (Kết hợp GĐ 1 & GĐ 2)",
            "value": effective_real,
            "color": "#10b981",
            "marker": None,
            "note": "Điểm người thật sau khi đối chiếu cả giả mạo vật lý lẫn Deepfake AI",
        },
        {
            "name": "Không giả mạo vật lý — GĐ 1 (Không phải giấy in / màn hình)",
            "value": stage1_real,
            "color": "#38bdf8",
            "marker": None,
            "note": "Chỉ đánh giá bề mặt vật lý (100% − Printed − On-screen)",
        },
        {
            "name": "Ảnh in giấy (Printed)",
            "value": probs.get(LABEL_PRINTED, 0.0),
            "color": "#f59e0b",
            "marker": None,
            "note": None,
        },
        {
            "name": "Qua màn hình (On-screen)",
            "value": probs.get(LABEL_ONSCREEN, 0.0),
            "color": "#f97316",
            "marker": None,
            "note": None,
        },
        {
            "name": "Tổng giả mạo vật lý (Spoof = Printed + On-screen)",
            "value": spoof_total,
            "color": "#fb923c",
            "marker": tau_spoof,
            "note": f"Vạch trắng là ngưỡng τ_spoof = {tau_spoof * 100:.0f}%",
        },
    ]

    if enable_deepfake:
        bars.append(
            {
                "name": "Khuôn mặt tổng hợp (Deepfake — GĐ 2)",
                "value": probs.get(LABEL_DEEPFAKE, 0.0),
                "color": "#ef4444",
                "marker": tau_deepfake,
                "note": f"Vạch trắng là ngưỡng τ_deepfake = {tau_deepfake * 100:.0f}%",
            }
        )

    chunks = ['<div class="prob-list">']
    for item in bars:
        val = max(0.0, min(1.0, float(item["value"])))
        pct = val * 100.0
        marker_html = ""
        if item["marker"] is not None:
            m_pct = max(0.0, min(100.0, float(item["marker"]) * 100.0))
            marker_html = f'<div class="prob-threshold-marker" style="left:{m_pct:.1f}%"></div>'
        note_html = f'<div class="prob-subnote">{html.escape(item["note"])}</div>' if item["note"] else ""
        chunks.append(
            f'<div class="prob-item">'
            f'<div class="prob-header"><span class="prob-label">{html.escape(item["name"])}</span>'
            f'<span class="prob-value">{pct:.1f}%</span></div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill" style="width:{pct:.1f}%;background:{item["color"]}"></div>'
            f'{marker_html}'
            f'</div>'
            f'{note_html}'
            f'</div>'
        )
    if not enable_deepfake:
        chunks.append(
            '<div class="prob-item"><div class="prob-header">'
            '<span class="prob-label">Deepfake (GĐ 2)</span>'
            '<span class="prob-value" style="color:#94a3b8">Đã tắt</span>'
            '</div><div class="prob-subnote">Bật công tắc Deepfake ở thanh bên để kích hoạt GĐ 2.</div></div>'
        )
    chunks.append("</div>")
    return "".join(chunks)


def _render_timeline_chart(
    timeline_rows: List[Dict[str, Any]],
    enable_deepfake: bool,
    tau_spoof: float,
    tau_deepfake: float,
) -> None:
    df = pd.DataFrame(timeline_rows)
    value_cols = [LABEL_REAL, LABEL_PRINTED, LABEL_ONSCREEN]
    if enable_deepfake and LABEL_DEEPFAKE in df.columns and df[LABEL_DEEPFAKE].notna().any():
        value_cols.append(LABEL_DEEPFAKE)

    melted = df.melt(
        id_vars=["frame_id", "timestamp_sec", "label", "latency_ms"],
        value_vars=value_cols,
        var_name="Nhánh",
        value_name="Xác suất",
    ).dropna(subset=["Xác suất"])

    color_scale = alt.Scale(
        domain=[LABEL_REAL, LABEL_PRINTED, LABEL_ONSCREEN, LABEL_DEEPFAKE],
        range=["#10b981", "#f59e0b", "#f97316", "#ef4444"],
    )

    base = alt.Chart(melted).encode(
        x=alt.X("frame_id:Q", title="Chỉ số khung hình (frame_id)", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Xác suất:Q", title="Xác suất", scale=alt.Scale(domain=[0.0, 1.0])),
        color=alt.Color("Nhánh:N", scale=color_scale, legend=alt.Legend(orient="top")),
        tooltip=[
            alt.Tooltip("frame_id:Q", title="Frame"),
            alt.Tooltip("timestamp_sec:Q", title="Thời điểm (s)"),
            alt.Tooltip("label:N", title="Nhãn frame"),
            alt.Tooltip("Nhánh:N", title="Nhánh"),
            alt.Tooltip("Xác suất:Q", title="Xác suất", format=".3f"),
            alt.Tooltip("latency_ms:Q", title="Độ trễ (ms)", format=".1f"),
        ],
    )

    layers = [
        base.mark_line(strokeWidth=2.4),
        base.mark_circle(size=65),
        alt.Chart(pd.DataFrame({"y": [tau_spoof], "Ngưỡng": [f"τ_spoof ({tau_spoof:.2f})"]}))
        .mark_rule(strokeDash=[5, 4], color="#fb923c", strokeWidth=1.5)
        .encode(y="y:Q"),
    ]
    if enable_deepfake:
        layers.append(
            alt.Chart(pd.DataFrame({"y": [tau_deepfake], "Ngưỡng": [f"τ_deepfake ({tau_deepfake:.2f})"]}))
            .mark_rule(strokeDash=[2, 4], color="#f87171", strokeWidth=1.5)
            .encode(y="y:Q")
        )

    chart = alt.layer(*layers).properties(height=300).interactive()
    st.altair_chart(chart, width="stretch")


def _render_person_card(f_item: Dict[str, Any], enable_deepfake: bool) -> None:
    """Hiển thị thẻ chi tiết cho một khuôn mặt được phát hiện trong khung cảnh."""
    p_idx = int(f_item.get("face_index", 1))
    p_lbl = str(f_item.get("label", LABEL_REAL))
    p_conf = float(f_item.get("confidence", 0.0))
    p_meta = LABEL_UI_META.get(p_lbl, LABEL_UI_META[LABEL_NO_FACE])
    s_probs = dict(f_item.get("spoof_probs") or {})
    f_prob = f_item.get("fake_prob")
    stage1_real = float(s_probs.get(LABEL_REAL, 0.0))
    eff_real = max(0.0, min(stage1_real, 1.0 - float(f_prob))) if (enable_deepfake and f_prob is not None) else stage1_real
    spoof_sum = float(s_probs.get(LABEL_PRINTED, 0.0)) + float(s_probs.get(LABEL_ONSCREEN, 0.0))
    df_txt = f"{float(f_prob) * 100:.1f}%" if (enable_deepfake and f_prob is not None) else "Tắt"

    with st.container(border=True):
        fc1, fc2 = st.columns([0.85, 1.45])
        crop_img = f_item.get("face_crop_rgb")
        if crop_img is not None:
            fc1.image(crop_img, caption=f"Người #{p_idx}", width="stretch")
        else:
            fc1.caption(f"Người #{p_idx}")
        with fc2:
            _render_html(
                f"""
                <span class="scene-verdict-pill" style="background:{p_meta['accent']}24;color:{p_meta['accent']};border:1px solid {p_meta['accent']}55;">
                    {p_meta['icon']} Người #{p_idx}: {html.escape(p_lbl)} ({p_conf * 100:.1f}%)
                </span>
                <div class="meta-chip-row">
                    <span class="meta-chip">Người thật: {eff_real * 100:.1f}%</span>
                    <span class="meta-chip">Spoof: {spoof_sum * 100:.1f}%</span>
                    <span class="meta-chip">Deepfake: {df_txt}</span>
                </div>
                """
            )


def _render_scene_gallery(result: PipelineExecutionResult, enable_deepfake: bool) -> None:
    """Hiển thị lưới các khung cảnh tách từ video/webcam và nhận diện toàn bộ từng người trong mỗi cảnh."""
    previews = list(getattr(result, "frame_previews", []) or [])
    if not previews:
        return

    with st.container(border=True):
        if len(previews) == 1:
            single_scene = previews[0]
            single_faces = list(single_scene.get("faces") or [])
            st.subheader(
                f"👥 Nhận Diện Toàn Bộ Mọi Người Trong Khung Hình ({len(single_faces)} khuôn mặt)"
            )
            if not single_faces:
                st.info("Không phát hiện khuôn mặt nào trong ảnh để phân tích chi tiết.")
                return
            for row_start in range(0, len(single_faces), 3):
                row_items = single_faces[row_start : row_start + 3]
                cols = st.columns(3)
                for col, f_item in zip(cols, row_items):
                    with col:
                        _render_person_card(f_item, enable_deepfake=enable_deepfake)
            return

        st.subheader(f"🎞️ Tách Khung Cảnh & Nhận Diện Theo Từng Đoạn ({len(previews)} khung hình)")

        total_cnt = len(previews)
        suspicious_cnt = sum(
            1
            for s in previews
            if str(s.get("frame_label")) in (LABEL_PRINTED, LABEL_ONSCREEN, LABEL_DEEPFAKE)
        )
        with_face_cnt = sum(1 for s in previews if bool(s.get("has_face")))
        no_face_cnt = total_cnt - with_face_cnt

        filter_options = {
            "Tất cả": f"Tất cả ({total_cnt})",
            "Cảnh nghi ngờ": f"Cảnh nghi ngờ ({suspicious_cnt})",
            "Có khuôn mặt": f"Có khuôn mặt ({with_face_cnt})",
            "Không thấy mặt": f"Không thấy mặt ({no_face_cnt})",
        }

        c_flt, c_cols = st.columns([2.2, 1.0])
        chosen_filter = c_flt.segmented_control(
            "Lọc khung cảnh",
            options=list(filter_options.keys()),
            format_func=lambda k: filter_options.get(k, k),
            key="scene_filter",
            selection_mode="single",
            required=True,
            label_visibility="collapsed",
        )
        chosen_cols = c_cols.selectbox(
            "Số cột lưới cảnh",
            options=["2 cột", "3 cột", "4 cột"],
            key="scene_grid_cols",
            label_visibility="collapsed",
        )
        n_cols = int(str(chosen_cols).split()[0])

        if chosen_filter == "Cảnh nghi ngờ":
            filtered = [
                s
                for s in previews
                if str(s.get("frame_label")) in (LABEL_PRINTED, LABEL_ONSCREEN, LABEL_DEEPFAKE)
            ]
        elif chosen_filter == "Có khuôn mặt":
            filtered = [s for s in previews if bool(s.get("has_face"))]
        elif chosen_filter == "Không thấy mặt":
            filtered = [s for s in previews if not bool(s.get("has_face"))]
        else:
            filtered = previews

        if not filtered:
            st.info("Không có khung cảnh nào khớp bộ lọc đang chọn.")
        else:
            for row_start in range(0, len(filtered), n_cols):
                row_items = filtered[row_start : row_start + n_cols]
                cols = st.columns(n_cols)
                for col, scene in zip(cols, row_items):
                    with col:
                        with st.container(border=True):
                            f_id = int(scene.get("frame_id", 0))
                            ts = float(scene.get("timestamp_sec", 0.0))
                            s_lbl = str(scene.get("frame_label", LABEL_NO_FACE))
                            s_conf = float(scene.get("frame_confidence", 0.0))
                            s_meta = LABEL_UI_META.get(s_lbl, LABEL_UI_META[LABEL_NO_FACE])
                            accent = s_meta["accent"]

                            _render_html(
                                f"""
                                <div class="scene-card-head">
                                    <span class="scene-time-pill">Cảnh #{f_id + 1} · ⏱ {ts:.2f}s</span>
                                    <span class="scene-verdict-pill" style="background:{accent}24;color:{accent};border:1px solid {accent}55;">
                                        {s_meta['icon']} {html.escape(s_lbl)} ({s_conf * 100:.0f}%)
                                    </span>
                                </div>
                                """
                            )

                            faces = list(scene.get("faces") or [])
                            frame_rgb = scene.get("frame_rgb")
                            if frame_rgb is not None:
                                annotated_scene = draw_multi_face_annotations(
                                    frame_rgb=frame_rgb,
                                    faces=faces,
                                )
                                st.image(annotated_scene, width="stretch")

                            if faces:
                                pills_html = ['<div class="scene-face-pills">']
                                for f_item in faces:
                                    p_lbl = str(f_item.get("label", LABEL_REAL))
                                    p_conf = float(f_item.get("confidence", 0.0))
                                    p_idx = int(f_item.get("face_index", 1))
                                    p_acc = LABEL_UI_META.get(p_lbl, LABEL_UI_META[LABEL_NO_FACE])["accent"]
                                    pills_html.append(
                                        f'<span class="person-mini-tag" style="border-color:{p_acc}66;">'
                                        f'<span style="color:{p_acc}">●</span> #{p_idx}: {html.escape(p_lbl)} ({p_conf * 100:.1f}%)'
                                        f'</span>'
                                    )
                                pills_html.append("</div>")
                                _render_html("".join(pills_html))
                            else:
                                st.caption("Không phát hiện khuôn mặt trong cảnh này.")

        st.divider()
        st.markdown("**🔎 Soi chi tiết từng khung cảnh & từng người xuất hiện**")
        scene_indices = list(range(len(previews)))

        def _format_scene_opt(idx: int) -> str:
            sc = previews[idx]
            f_cnt = len(sc.get("faces") or [])
            return (
                f"Cảnh #{int(sc.get('frame_id', idx)) + 1} "
                f"(⏱ {float(sc.get('timestamp_sec', 0.0)):.2f}s) — "
                f"{sc.get('frame_label', LABEL_NO_FACE)} — {f_cnt} khuôn mặt"
            )

        selected_idx = st.selectbox(
            "Chọn khung cảnh để phóng to kiểm tra từng khuôn mặt",
            options=scene_indices,
            format_func=_format_scene_opt,
        )
        chosen_scene = previews[int(selected_idx)]
        chosen_faces = list(chosen_scene.get("faces") or [])

        mag_left, mag_right = st.columns([1.15, 1.0], gap="large")
        with mag_left:
            frm = chosen_scene.get("frame_rgb")
            if frm is not None:
                annotated_mag = draw_multi_face_annotations(frm, chosen_faces)
                st.image(
                    annotated_mag,
                    caption=(
                        f"Cảnh #{int(chosen_scene.get('frame_id', 0)) + 1} "
                        f"tại {float(chosen_scene.get('timestamp_sec', 0.0)):.2f}s"
                    ),
                    width="stretch",
                )
        with mag_right:
            if not chosen_faces:
                st.info("Không phát hiện khuôn mặt nào trong khung cảnh này.")
            else:
                for f_item in chosen_faces:
                    _render_person_card(f_item, enable_deepfake=enable_deepfake)


def _render_results(
    result: PipelineExecutionResult,
    tau_spoof: float,
    tau_deepfake: float,
    uncertain_margin: float,
    enable_deepfake: bool,
    meta: Dict[str, Any],
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

    col_left, col_right = st.columns([1.02, 1.18], gap="large")

    with col_left:
        with st.container(border=True):
            st.subheader("Kiểm tra thị giác khung hình")
            if result.preview_frame_rgb is not None:
                previews_list = list(getattr(result, "frame_previews", []) or [])
                first_face_preview = next((p for p in previews_list if p.get("faces")), None)
                rep_faces = list(first_face_preview.get("faces") or []) if first_face_preview else []
                rep_frame = (
                    first_face_preview.get("frame_rgb")
                    if first_face_preview and first_face_preview.get("frame_rgb") is not None
                    else result.preview_frame_rgb
                )
                if rep_faces:
                    annotated = draw_multi_face_annotations(
                        frame_rgb=rep_frame,
                        faces=rep_faces,
                    )
                else:
                    annotated = draw_face_annotation(
                        frame_rgb=result.preview_frame_rgb,
                        bbox=result.preview_bbox,
                        label=result.final_label,
                        confidence=result.confidence,
                    )
                tab_hud, tab_crop, tab_cmp = st.tabs(
                    ["🎯 Khung hình đánh dấu", "👤 Khuôn mặt chuẩn M3", "🔀 So sánh Gốc / HUD"]
                )
                with tab_hud:
                    st.image(annotated, caption="Khung hình đại diện kèm bounding box HUD", width="stretch")
                    if result.preview_bbox is not None:
                        x1, y1, x2, y2 = result.preview_bbox
                        _render_html(
                            f"""
                            <div class="meta-chip-row">
                                <span class="meta-chip">BBox chính: ({x1}, {y1}) → ({x2}, {y2})</span>
                                <span class="meta-chip">Kích thước mặt: {max(0, x2 - x1)}×{max(0, y2 - y1)} px</span>
                                <span class="meta-chip">Số mặt trong khung: {max(1, len(rep_faces))}</span>
                            </div>
                            """
                        )
                with tab_crop:
                    cropped_items = [f for f in rep_faces if f.get("face_crop_rgb") is not None]
                    if len(cropped_items) > 1:
                        for r_start in range(0, len(cropped_items), 4):
                            row_faces = cropped_items[r_start : r_start + 4]
                            crop_cols = st.columns(len(row_faces))
                            for c_col, c_face in zip(crop_cols, row_faces):
                                c_lbl = str(c_face.get("label", LABEL_REAL))
                                c_conf = float(c_face.get("confidence", 0.0))
                                c_idx = int(c_face.get("face_index", 1))
                                c_col.image(
                                    c_face["face_crop_rgb"],
                                    caption=f"Người #{c_idx}: {c_lbl} ({c_conf * 100:.1f}%)",
                                    width="stretch",
                                )
                    elif result.preview_face_crop_rgb is not None:
                        st.image(
                            result.preview_face_crop_rgb,
                            caption=(
                                f"FaceCropResult.face_rgb — {result.preview_face_crop_rgb.shape} "
                                f"({result.preview_face_crop_rgb.dtype})"
                            ),
                            width=240,
                        )
                    else:
                        st.info("Chưa có ảnh khuôn mặt đã cắt (không phát hiện mặt hoặc không trả về crop).")
                with tab_cmp:
                    c_orig, c_hud = st.columns(2)
                    c_orig.image(rep_frame, caption="Ảnh gốc đầu vào", width="stretch")
                    c_hud.image(annotated, caption="Sau phân tích HUD", width="stretch")
            else:
                st.info("Không có khung hình đại diện để hiển thị.")

    with col_right:
        with st.container(border=True):
            st.subheader("Kết luận & Xác suất đa nhánh")
            display_label = LABEL_DISPLAY_VI.get(result.final_label, result.final_label)
            ui_meta = LABEL_UI_META.get(result.final_label, LABEL_UI_META[LABEL_NO_FACE])
            card_mod = _verdict_css_modifier(result.final_label)
            ring_svg = _build_confidence_ring_svg(result.confidence, ui_meta["accent"])
            reason_safe = html.escape(result.reason or ui_meta["summary"])

            _render_html(
                f"""
                <div class="verdict-card {card_mod}">
                    <div>
                        <span class="verdict-badge" style="background:{ui_meta['accent']}24;color:{ui_meta['accent']};border:1px solid {ui_meta['accent']}55;">
                            {ui_meta['icon']} {html.escape(ui_meta['badge_text'])}
                        </span>
                        <div class="verdict-title">{html.escape(display_label)}</div>
                        <p class="verdict-meta">{reason_safe}</p>
                    </div>
                    <div class="verdict-ring-wrap">
                        {ring_svg}
                        <span class="verdict-ring-caption">Độ tin cậy</span>
                    </div>
                </div>
                """
            )

            _render_html(_render_prob_bars_html(probs, tau_spoof, tau_deepfake, enable_deepfake))

            perf = compute_performance_stats(result.frame_logs, result.elapsed_sec)
            _render_html(
                f"""
                <div class="kpi-grid">
                    <div class="kpi-box">
                        <p class="kpi-label">Số frame xử lý</p>
                        <p class="kpi-val">{int(perf['num_frames'])}</p>
                    </div>
                    <div class="kpi-box">
                        <p class="kpi-label">Tốc độ (FPS)</p>
                        <p class="kpi-val">{perf['effective_fps']:.2f}</p>
                    </div>
                    <div class="kpi-box">
                        <p class="kpi-label">Độ trễ TB</p>
                        <p class="kpi-val">{perf['avg_latency_ms']:.1f} ms</p>
                    </div>
                    <div class="kpi-box">
                        <p class="kpi-label">Tổng thời gian</p>
                        <p class="kpi-val">{perf['total_elapsed_sec']:.2f} s</p>
                    </div>
                </div>
                """
            )

    st.markdown("")
    _render_scene_gallery(result=result, enable_deepfake=enable_deepfake)

    st.markdown("")
    with st.container(border=True):
        st.subheader("Phân tích chi tiết & Báo cáo phiên")
        timeline_rows = build_frame_timeline_rows(result.frame_logs)
        tab_chart, tab_table, tab_export = st.tabs(
            ["📈 Biểu đồ dòng thời gian Frame", "📋 Log chi tiết từng Frame", "📥 Lịch sử & Xuất báo cáo"]
        )

        with tab_chart:
            if timeline_rows:
                _render_timeline_chart(
                    timeline_rows=timeline_rows,
                    enable_deepfake=enable_deepfake,
                    tau_spoof=tau_spoof,
                    tau_deepfake=tau_deepfake,
                )
            else:
                st.caption("Chưa có dữ liệu dòng thời gian frame từ pipeline.")

        with tab_table:
            if timeline_rows:
                st.dataframe(
                    timeline_rows,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "frame_id": st.column_config.NumberColumn("Frame", format="%d"),
                        "timestamp_sec": st.column_config.NumberColumn("Giây", format="%.2f"),
                        "label": st.column_config.TextColumn("Nhãn"),
                        "Real": st.column_config.ProgressColumn("Real", min_value=0.0, max_value=1.0, format="%.3f"),
                        "Printed": st.column_config.ProgressColumn("Printed", min_value=0.0, max_value=1.0, format="%.3f"),
                        "On-screen": st.column_config.ProgressColumn("On-screen", min_value=0.0, max_value=1.0, format="%.3f"),
                        "Deepfake": st.column_config.ProgressColumn("Deepfake", min_value=0.0, max_value=1.0, format="%.3f"),
                        "latency_ms": st.column_config.NumberColumn("Độ trễ (ms)", format="%.1f"),
                        "has_face": st.column_config.CheckboxColumn("Có mặt"),
                    },
                )
            else:
                st.caption("Chưa có log chi tiết từng frame được trả về từ pipeline.")

        with tab_export:
            cfg = meta.get("config") or {}
            report_dict = build_export_report_dict(
                result=result,
                config=cfg,
                input_mode=str(meta.get("input_mode", "Unknown")),
                source_name=str(meta.get("source_name", "session_input")),
            )
            json_bytes = json.dumps(report_dict, ensure_ascii=False, indent=2).encode("utf-8")
            csv_bytes = (
                pd.DataFrame(timeline_rows).to_csv(index=False).encode("utf-8-sig")
                if timeline_rows
                else b"frame_id,label\n"
            )
            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "Tải báo cáo tổng hợp (.json)",
                data=json_bytes,
                file_name="antideepfake_report.json",
                mime="application/json",
                on_click="ignore",
                width="stretch",
            )
            dl2.download_button(
                "Tải bảng log từng frame (.csv)",
                data=csv_bytes,
                file_name="antideepfake_timeline.csv",
                mime="text/csv",
                on_click="ignore",
                width="stretch",
            )
            history = st.session_state.get("analysis_history", [])
            if history:
                st.caption("Lịch sử phân tích trong phiên RAM hiện tại (tự xóa khi đóng phiên):")
                st.dataframe(history, width="stretch", hide_index=True)


def _render_image_workspace(
    config: Dict[str, Any],
    backend_available: bool,
    allow_ui_preview: bool,
) -> None:
    col_upload, col_preview = st.columns([1.05, 1.15], gap="large")

    with col_upload:
        uploaded = st.file_uploader(
            "Chọn ảnh khuôn mặt (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help=f"Dung lượng tối đa {MAX_IMAGE_SIZE_MB} MB. Hỗ trợ JPG, JPEG, PNG.",
        )
        if _SAMPLE_FACE_PATH.is_file():
            c_demo, c_clr = st.columns([1.4, 0.8])
            if c_demo.button("✨ Dùng ảnh mẫu thử nghiệm", width="stretch"):
                st.session_state["demo_image_bytes"] = _SAMPLE_FACE_PATH.read_bytes()
                st.session_state["demo_image_name"] = _SAMPLE_FACE_PATH.name
            if st.session_state.get("demo_image_bytes") is not None:
                if c_clr.button("Bỏ ảnh mẫu", width="stretch"):
                    st.session_state["demo_image_bytes"] = None
                    st.session_state["demo_image_name"] = None
                    st.rerun()

        active_bytes: Optional[bytes] = None
        active_name: str = ""
        if uploaded is not None:
            active_bytes = uploaded.getvalue()
            active_name = uploaded.name
        elif st.session_state.get("demo_image_bytes") is not None:
            active_bytes = st.session_state["demo_image_bytes"]
            active_name = str(st.session_state.get("demo_image_name") or "sample_face.jpg")

        if active_bytes is None:
            _render_html(
                """
                <div class="empty-guide">
                    <p class="empty-guide-title">Hướng dẫn kiểm tra ảnh tĩnh</p>
                    <ul>
                        <li>Tải lên ảnh chụp chân dung rõ mặt (JPG/PNG) hoặc nhấn <strong>Dùng ảnh mẫu thử nghiệm</strong>.</li>
                        <li>Tránh ảnh bị che khuất, quá tối hoặc độ phân giải quá thấp.</li>
                        <li>Hệ thống chỉ xử lý trong bộ nhớ tạm và xóa ngay sau khi phân tích xong.</li>
                    </ul>
                </div>
                """
            )

    with col_preview:
        if active_bytes is not None:
            val = validate_image_input(active_name, active_bytes)
            if not val.is_valid:
                st.error(val.error_message or UI_ERRORS["invalid_file"])
                return

            rgb_img = _decode_rgb_image(active_bytes)
            st.image(rgb_img, caption=f"Xem trước: {active_name}", width="stretch")
            _render_html(
                f"""
                <div class="meta-chip-row">
                    <span class="meta-chip">{val.width}×{val.height} px</span>
                    <span class="meta-chip">{val.size_mb:.2f} MB</span>
                    <span class="meta-chip">τ_spoof={config['thresholds']['tau_spoof']:.2f}</span>
                    <span class="meta-chip">Deepfake={'ON' if config['enable_deepfake'] else 'OFF'}</span>
                </div>
                """
            )
            if st.button("🔍 Phân tích khuôn mặt", type="primary", width="stretch"):
                suffix = Path(active_name).suffix.lower() or ".jpg"
                with st.spinner("Đang phân tích khuôn mặt qua các tầng mô hình..."):
                    res = _execute_file_analysis(
                        raw_bytes=active_bytes,
                        suffix=suffix,
                        config=config,
                        fallback_frame_rgb=rgb_img,
                        backend_available=backend_available,
                        allow_ui_preview=allow_ui_preview,
                        is_video=False,
                    )
                if res is not None:
                    _record_result(res, "Upload ảnh", active_name, config)
        else:
            st.info("Chọn ảnh ở cột bên trái để xem trước thông số và bắt đầu phân tích.")


def _render_video_workspace(
    config: Dict[str, Any],
    backend_available: bool,
    allow_ui_preview: bool,
) -> None:
    col_upload, col_preview = st.columns([1.05, 1.15], gap="large")

    with col_upload:
        uploaded = st.file_uploader(
            "Chọn video ngắn (MP4, AVI, MOV, MKV)",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=False,
        )
        mode_map = {
            "scene": "Tự động bắt cảnh (toàn video)",
            "uniform": "Chia đều toàn thời lượng",
            "fps": f"{config['sample_fps']} FPS cố định",
        }
        mode_txt = mode_map.get(str(config.get("sampling_mode", "scene")), "Scene")
        _render_html(
            f"""
            <div class="meta-chip-row">
                <span class="meta-chip">Dung lượng ≤ {MAX_VIDEO_SIZE_MB} MB</span>
                <span class="meta-chip">Thời lượng ≤ {MAX_VIDEO_DURATION_SEC}s</span>
                <span class="meta-chip">Tối đa {config['max_frames']} cảnh · {mode_txt}</span>
            </div>
            """
        )
        if uploaded is None:
            _render_html(
                """
                <div class="empty-guide">
                    <p class="empty-guide-title">Tách khung cảnh &amp; Nhận diện toàn video</p>
                    <ul>
                        <li>Chế độ <strong>Tự động bắt từng khung cảnh (Scene)</strong> quét toàn bộ video để chọn tối đa <strong>N</strong> cảnh khác biệt nhất.</li>
                        <li>Tất cả các khuôn mặt xuất hiện trong từng cảnh sẽ được đóng khung, phân loại riêng và hiển thị thành lưới Storyboard bên dưới.</li>
                    </ul>
                </div>
                """
            )

    with col_preview:
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
                return

            st.video(raw_bytes)
            dur_txt = f"{val.duration_sec:.1f}s" if val.duration_sec is not None else "N/A"
            dim_txt = f"{val.width}×{val.height}" if val.width and val.height else "N/A"
            _render_html(
                f"""
                <div class="meta-chip-row">
                    <span class="meta-chip">{val.size_mb:.2f} MB</span>
                    <span class="meta-chip">Thời lượng: {dur_txt}</span>
                    <span class="meta-chip">Độ phân giải: {dim_txt}</span>
                </div>
                """
            )
            if st.button("🎬 Phân tích chuỗi khung hình", type="primary", width="stretch"):
                res = _execute_file_analysis(
                    raw_bytes=raw_bytes,
                    suffix=suffix,
                    config=config,
                    fallback_frame_rgb=None,
                    backend_available=backend_available,
                    allow_ui_preview=allow_ui_preview,
                    is_video=True,
                )
                if res is not None:
                    _record_result(res, "Upload video", uploaded.name, config)
        else:
            st.info("Tải lên tệp video để xem trước và kích hoạt nút phân tích.")


@st.fragment(run_every=1.5)
def _live_opencv_hud_fragment(
    device_id: int,
    config: Dict[str, Any],
    backend_available: bool,
    allow_ui_preview: bool,
) -> None:
    """Fragment tự động chụp & cập nhật HUD Webcam mỗi 1.5 giây theo thời gian thực."""
    burst = capture_webcam_burst(device_id=device_id, num_frames=1)
    if not burst.success or not burst.frames_rgb:
        st.error(burst.error_message or UI_ERRORS["webcam_unavailable"])
        st.session_state["live_webcam_active"] = False
        return

    buffer: SlidingFrameBuffer = st.session_state["webcam_buffer"]
    buffer.push(burst.frames_rgb[0])
    recent = buffer.get_frames()[-min(5, len(buffer)) :]
    live_cfg = {**config, "max_frames": len(recent), "sampling_mode": "fps"}

    if backend_available:
        live_res = run_backend_frames_pipeline(recent, live_cfg)
    elif allow_ui_preview:
        live_res = run_ui_preview_simulation(recent, live_cfg)
    else:
        st.warning("Pipeline chưa sẵn sàng cho chế độ Live.")
        return

    previews = list(getattr(live_res, "frame_previews", []) or [])
    last_pv = previews[-1] if previews else None
    disp_frame = last_pv.get("frame_rgb") if (last_pv and last_pv.get("frame_rgb") is not None) else burst.frames_rgb[0]
    disp_faces = list(last_pv.get("faces") or []) if last_pv else []
    annotated = draw_multi_face_annotations(
        disp_frame,
        disp_faces,
        fallback_bbox=live_res.preview_bbox,
        fallback_label=live_res.final_label,
        fallback_confidence=live_res.confidence,
    )

    c_cam, c_stat = st.columns([1.2, 1.0], gap="large")
    with c_cam:
        st.image(annotated, caption=f"Live HUD (Device ID {device_id} · Bộ đệm {len(buffer)} frame)", width="stretch")
    with c_stat:
        ui_meta = LABEL_UI_META.get(live_res.final_label, LABEL_UI_META[LABEL_NO_FACE])
        probs = normalize_aggregated_probs(live_res.aggregated_probs)
        spoof_p = (probs[LABEL_PRINTED] + probs[LABEL_ONSCREEN]) * 100.0
        _render_html(
            f"""
            <div class="verdict-card {_verdict_css_modifier(live_res.final_label)}">
                <div>
                    <span class="verdict-badge" style="background:{ui_meta['accent']}24;color:{ui_meta['accent']};border:1px solid {ui_meta['accent']}55;">
                        🔴 LIVE · {html.escape(ui_meta['badge_text'])}
                    </span>
                    <div class="verdict-title">{html.escape(LABEL_DISPLAY_VI.get(live_res.final_label, live_res.final_label))}</div>
                    <p class="verdict-meta">{html.escape(live_res.reason or ui_meta['summary'])}</p>
                </div>
                <div class="verdict-ring-wrap">
                    {_build_confidence_ring_svg(live_res.confidence, ui_meta['accent'])}
                </div>
            </div>
            <div class="meta-chip-row">
                <span class="meta-chip">Real: {probs[LABEL_REAL]*100:.1f}%</span>
                <span class="meta-chip">Spoof: {spoof_p:.1f}%</span>
                <span class="meta-chip">Deepfake: {probs[LABEL_DEEPFAKE]*100:.1f}%</span>
            </div>
            """
        )


def _render_webcam_workspace(
    config: Dict[str, Any],
    backend_available: bool,
    allow_ui_preview: bool,
) -> None:
    webcam_mode = st.radio(
        "Phương thức đọc webcam",
        options=["Camera trình duyệt (st.camera_input)", "Webcam cục bộ (OpenCV ID 0)"],
        horizontal=True,
    )
    buffer: SlidingFrameBuffer = st.session_state["webcam_buffer"]
    n_frames = int(config["max_frames"])

    if webcam_mode == "Camera trình duyệt (st.camera_input)":
        col_cam, col_buf = st.columns([1.1, 1.0], gap="large")
        with col_cam:
            cam_photo = st.camera_input("Chụp ảnh từ webcam (hỗ trợ chụp nhiều lần để gộp N frame)")
            if cam_photo is not None:
                raw_bytes = cam_photo.getvalue()
                digest = hashlib.sha256(raw_bytes).hexdigest()
                if digest != st.session_state.get("last_cam_digest"):
                    rgb_cam = _decode_rgb_image(raw_bytes)
                    buffer.push(rgb_cam)
                    st.session_state["last_cam_digest"] = digest

        with col_buf:
            buf_len = len(buffer)
            _render_html(
                f"""
                <div class="buffer-badge">
                    <strong>Bộ đệm khung hình trượt (DD1):</strong>
                    {buf_len} / {DEFAULT_MAX_FRAMES} frame đã lưu
                    (sẽ dùng tối đa {min(buf_len, n_frames) if buf_len else n_frames} frame khi phân tích)
                </div>
                """
            )
            frames = buffer.get_frames()
            if frames:
                preview_slice = frames[-4:]
                thumb_cols = st.columns(len(preview_slice))
                start_idx = max(1, len(frames) - len(preview_slice) + 1)
                for offset, (col, frm) in enumerate(zip(thumb_cols, preview_slice)):
                    col.image(frm, caption=f"#{start_idx + offset}", width="stretch")
            else:
                st.caption("Chưa có khung hình nào trong bộ đệm. Hãy chụp ảnh từ camera bên trái.")

            c_run, c_clear = st.columns([1.4, 0.8])
            run_clicked = c_run.button(
                "🔍 Phân tích bộ đệm webcam",
                type="primary",
                width="stretch",
                disabled=buf_len == 0,
            )
            if c_clear.button("🗑️ Xóa bộ đệm webcam", width="stretch", disabled=buf_len == 0):
                buffer.clear()
                st.session_state["last_cam_digest"] = None
                st.rerun()

            if run_clicked and frames:
                with st.spinner("Đang phân tích chuỗi khung hình webcam..."):
                    if not backend_available and allow_ui_preview:
                        res = run_ui_preview_simulation(frames, config)
                    elif backend_available:
                        res = run_backend_frames_pipeline(frames, config)
                    else:
                        st.error(
                            "Pipeline `src/pipeline.py` chưa sẵn sàng. "
                            "Hãy bật 'Cho phép chạy thử giao diện' ở thanh bên để xem thử UI."
                        )
                        res = None
                if res is not None:
                    _record_result(res, "Webcam (Trình duyệt)", f"camera_buffer ({len(frames)} frame)", config)
    else:
        col_dev, col_act = st.columns([1.0, 1.2], gap="large")
        with col_dev:
            device_id = int(st.number_input("Webcam Device ID", min_value=0, max_value=5, value=0, step=1))
            if st.button("📡 Kiểm tra kết nối camera", width="stretch"):
                ok_cam, cam_err = check_webcam_available(device_id=device_id)
                st.session_state["webcam_probe_status"] = (ok_cam, cam_err)
            probe = st.session_state.get("webcam_probe_status")
            if probe is not None:
                if probe[0]:
                    st.success(f"Webcam ID {device_id} sẵn sàng hoạt động.")
                else:
                    st.error(probe[1] or UI_ERRORS["webcam_unavailable"])

        with col_act:
            _render_html(
                f"""
                <div class="empty-guide">
                    <p class="empty-guide-title">Thu luồng trực tiếp qua OpenCV</p>
                    <ul>
                        <li>Chỉ khả dụng khi chạy server cục bộ trên máy có thiết bị camera gắn trực tiếp.</li>
                        <li>Có thể chụp 1 đợt <strong>{n_frames}</strong> frame hoặc bật <strong>Quét Live</strong> tự động cập nhật khung HUD mỗi 1.5s.</li>
                    </ul>
                </div>
                """
            )
            st.markdown("")
            c_burst, c_live = st.columns(2)
            if c_burst.button("🎥 Thu đợt & Phân tích", type="primary", width="stretch"):
                st.session_state["live_webcam_active"] = False
                ok_cam, cam_err = check_webcam_available(device_id=device_id)
                if not ok_cam:
                    st.error(cam_err or UI_ERRORS["webcam_unavailable"])
                elif backend_available:
                    with st.spinner("Đang đọc webcam và chạy pipeline..."):
                        res = run_backend_pipeline(device_id, config)
                    _record_result(res, "Webcam (OpenCV)", f"device_id={device_id}", config)
                elif allow_ui_preview:
                    burst = capture_webcam_burst(device_id=device_id, num_frames=min(n_frames, 5))
                    if not burst.success:
                        st.error(burst.error_message or UI_ERRORS["webcam_unavailable"])
                    else:
                        res = run_ui_preview_simulation(burst.frames_rgb, config)
                        _record_result(res, "Webcam (OpenCV Preview)", f"device_id={device_id}", config)
                else:
                    st.error(UI_ERRORS["webcam_unavailable"])

            live_on = bool(st.session_state.get("live_webcam_active"))
            live_lbl = "⏹️ Dừng Quét Live" if live_on else "🔴 Bật Quét Live (1.5s)"
            if c_live.button(live_lbl, width="stretch"):
                if not live_on:
                    ok_cam, cam_err = check_webcam_available(device_id=device_id)
                    if not ok_cam:
                        st.error(cam_err or UI_ERRORS["webcam_unavailable"])
                    else:
                        st.session_state["live_webcam_active"] = True
                        st.rerun()
                else:
                    st.session_state["live_webcam_active"] = False
                    st.rerun()

        if st.session_state.get("live_webcam_active"):
            st.divider()
            _live_opencv_hud_fragment(
                device_id=device_id,
                config=config,
                backend_available=backend_available,
                allow_ui_preview=allow_ui_preview,
            )


def main() -> None:
    st.set_page_config(
        page_title="AntiDeepFake-Face | Phát hiện gian lận khuôn mặt",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _load_css()
    _install_dom_translation_guard()
    _init_session_state()

    backend_status = get_cached_backend_status()
    backend_available = bool(backend_status["available"])

    allow_ui_preview = _render_sidebar(backend_available=backend_available)

    config = build_pipeline_config(
        sample_fps=int(st.session_state["sample_fps"]),
        max_frames=int(st.session_state["n_frames"]),
        tau_spoof=float(st.session_state["tau_spoof"]),
        tau_deepfake=float(st.session_state["tau_deepfake"]),
        uncertain_margin=float(st.session_state["uncertain_margin"]),
        enable_deepfake=bool(st.session_state["enable_deepfake"]),
        deepfake_variant=str(st.session_state["deepfake_variant"]),
        sampling_mode=str(st.session_state["sampling_mode"]),
        multi_face=bool(st.session_state["multi_face"]),
        include_frame_previews=True,
        aggregation_strategy=str(st.session_state["aggregation_strategy"]),
        max_faces_per_frame=int(st.session_state["max_faces_per_frame"]),
        det_conf_threshold=float(st.session_state["det_conf_threshold"]),
    )

    latest_result: Optional[PipelineExecutionResult] = st.session_state.get("latest_result")
    _render_hero_and_stepper(
        backend_available=backend_available,
        enable_deepfake=bool(st.session_state["enable_deepfake"]),
        latest_result=latest_result,
    )

    with st.container(border=True):
        st.segmented_control(
            "Chế độ đầu vào",
            options=_INPUT_MODES,
            key="main_mode_ctrl",
            on_change=_sync_mode_from_main,
            selection_mode="single",
            required=True,
            width="stretch",
            label_visibility="collapsed",
        )
        active_mode = st.session_state["active_mode"]
        if active_mode == "Upload ảnh":
            _render_image_workspace(
                config=config,
                backend_available=backend_available,
                allow_ui_preview=allow_ui_preview,
            )
        elif active_mode == "Upload video":
            _render_video_workspace(
                config=config,
                backend_available=backend_available,
                allow_ui_preview=allow_ui_preview,
            )
        else:
            _render_webcam_workspace(
                config=config,
                backend_available=backend_available,
                allow_ui_preview=allow_ui_preview,
            )

    latest_result = st.session_state.get("latest_result")
    if latest_result is not None:
        st.markdown("")
        _render_results(
            result=latest_result,
            tau_spoof=float(st.session_state["tau_spoof"]),
            tau_deepfake=float(st.session_state["tau_deepfake"]),
            uncertain_margin=float(st.session_state["uncertain_margin"]),
            enable_deepfake=bool(st.session_state["enable_deepfake"]),
            meta=dict(st.session_state.get("latest_meta") or {}),
        )


if __name__ == "__main__":
    main()
