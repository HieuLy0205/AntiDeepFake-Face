"""Cầu nối giao tiếp giữa Streamlit UI (M6) và src/pipeline.py (M1)."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image

_ADF_CODE_REV = "2026.11.v5"

from app.components.constants import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_SAMPLE_FPS,
    DEFAULT_TAU_DEEPFAKE,
    DEFAULT_TAU_SPOOF,
    DEFAULT_UNCERTAIN_MARGIN,
    LABEL_DEEPFAKE,
    LABEL_NO_FACE,
    LABEL_ONSCREEN,
    LABEL_PRINTED,
    LABEL_REAL,
    UI_ERRORS,
)


def ensure_project_root_on_path() -> Path:
    """Đảm bảo thư mục gốc repo nằm trong sys.path khi chạy `streamlit run app/app.py`."""
    project_root = Path(__file__).resolve().parents[2]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


@dataclass
class PipelineExecutionResult:
    """Kết quả chuẩn hóa trả về cho trang Streamlit."""

    final_label: str
    confidence: float
    aggregated_probs: Dict[str, float]
    reason: str
    frame_logs: List[dict] = field(default_factory=list)
    preview_frame_rgb: Optional[np.ndarray] = None
    preview_bbox: Optional[Tuple[int, int, int, int]] = None
    preview_face_crop_rgb: Optional[np.ndarray] = None
    frame_previews: List[dict] = field(default_factory=list)
    elapsed_sec: float = 0.0
    is_mock_preview: bool = False
    error_message: Optional[str] = None


def build_pipeline_config(
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    tau_spoof: float = DEFAULT_TAU_SPOOF,
    tau_deepfake: float = DEFAULT_TAU_DEEPFAKE,
    uncertain_margin: float = DEFAULT_UNCERTAIN_MARGIN,
    enable_deepfake: bool = True,
    deepfake_variant: str = "siglip2",
    sampling_mode: str = "fps",
    multi_face: bool = False,
    include_frame_previews: bool = False,
    aggregation_strategy: str = "primary",
    max_faces_per_frame: int = 20,
    det_conf_threshold: float = 0.62,
    **extra_config: Any,
) -> Dict[str, Any]:
    """Tạo dict config đồng bộ cho `src.pipeline.run_pipeline(source, config)`."""
    cfg: Dict[str, Any] = {
        "sample_fps": int(sample_fps),
        "max_frames": int(max_frames),
        "enable_deepfake": bool(enable_deepfake),
        "deepfake_variant": str(deepfake_variant),
        "sampling_mode": str(sampling_mode),
        "multi_face": bool(multi_face),
        "include_frame_previews": bool(include_frame_previews),
        "aggregation_strategy": str(aggregation_strategy),
        "max_faces_per_frame": int(max_faces_per_frame),
        "det_conf_threshold": float(det_conf_threshold),
        "thresholds": {
            "tau_spoof": float(tau_spoof),
            "tau_deepfake": float(tau_deepfake),
            "uncertain_margin": float(uncertain_margin),
            # Alias dự phòng nếu M1 đọc key ngắn
            "spoof": float(tau_spoof),
            "deepfake": float(tau_deepfake),
        },
    }
    if extra_config:
        cfg.update(extra_config)
    return cfg


def query_deepfake_model_status(variant: str = "siglip2") -> Dict[str, Any]:
    """Kiểm tra trạng thái adapter và cache model M5 từ phía UI."""
    ensure_project_root_on_path()
    try:
        mod = importlib.import_module("src.deepfake.predictor")
        fn = getattr(mod, "get_deepfake_runtime_status", None)
        if callable(fn):
            return dict(fn(variant))
    except Exception:
        pass
    return {
        "variant": variant,
        "has_siglip_adapter": False,
        "has_vit_adapter": False,
        "has_cnn_checkpoint": False,
        "siglip_cached": False,
        "vit_cached": False,
    }


def is_backend_pipeline_available() -> Tuple[bool, Optional[str]]:
    """Kiểm tra xem `src.pipeline.run_pipeline` đã được M1 tích hợp chưa."""
    ensure_project_root_on_path()
    try:
        module = importlib.import_module("src.pipeline")
        if hasattr(module, "run_pipeline"):
            return True, None
        return False, "Module `src.pipeline` chưa có hàm `run_pipeline`."
    except Exception as exc:
        return False, str(exc)


@contextmanager
def temporary_memory_file(raw_bytes: bytes, suffix: str) -> Iterator[str]:
    """Tạo tệp tạm trong RAM/temp và xóa ngay sau khi xử lý xong (DD8: không lưu dữ liệu)."""
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="adf_session_")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw_bytes)
        yield temp_path
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


def _extract_preview_from_decision(
    decision: Any,
    fallback_frame_rgb: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Trích xuất khung hình đại diện và bounding box từ FinalDecision hoặc frame đầu."""
    bbox = getattr(decision, "preview_bbox", None)
    preview_rgb = getattr(decision, "preview_frame_rgb", None)

    frame_logs = getattr(decision, "frame_logs", []) or []
    if bbox is None:
        for entry in frame_logs:
            if isinstance(entry, dict) and entry.get("bbox") is not None:
                raw_bbox = entry["bbox"]
                if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                    bbox = tuple(int(v) for v in raw_bbox)
                    break

    if preview_rgb is None:
        for entry in frame_logs:
            if isinstance(entry, dict) and entry.get("frame_rgb") is not None:
                preview_rgb = entry["frame_rgb"]
                break

    if preview_rgb is None:
        preview_rgb = fallback_frame_rgb

    return preview_rgb, bbox  # type: ignore


def run_backend_pipeline(
    source: Union[str, int, Sequence[np.ndarray]],
    config: Dict[str, Any],
    fallback_frame_rgb: Optional[np.ndarray] = None,
) -> PipelineExecutionResult:
    """Gọi `src.pipeline.run_pipeline(source, config)` và chuẩn hóa ngoại lệ theo II.6."""
    ensure_project_root_on_path()
    start = time.perf_counter()

    try:
        pipeline_mod = importlib.import_module("src.pipeline")
        run_pipeline_fn = getattr(pipeline_mod, "run_pipeline")
        decision = run_pipeline_fn(source, config)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        err_name = type(exc).__name__
        err_msg = str(exc)
        if err_name == "InvalidInputError":
            ui_msg = UI_ERRORS["invalid_file"]
        else:
            ui_msg = f"Lỗi xử lý pipeline ({err_name}): {err_msg}"
        return PipelineExecutionResult(
            final_label=LABEL_NO_FACE,
            confidence=0.0,
            aggregated_probs={
                LABEL_REAL: 0.0,
                LABEL_PRINTED: 0.0,
                LABEL_ONSCREEN: 0.0,
                LABEL_DEEPFAKE: 0.0,
            },
            reason=ui_msg,
            elapsed_sec=elapsed,
            error_message=ui_msg,
        )

    elapsed = time.perf_counter() - start
    preview_rgb, bbox = _extract_preview_from_decision(decision, fallback_frame_rgb)

    return PipelineExecutionResult(
        final_label=str(getattr(decision, "final_label", LABEL_NO_FACE)),
        confidence=float(getattr(decision, "confidence", 0.0)),
        aggregated_probs=dict(getattr(decision, "aggregated_probs", {}) or {}),
        reason=str(getattr(decision, "reason", "")),
        frame_logs=list(getattr(decision, "frame_logs", []) or []),
        preview_frame_rgb=preview_rgb,
        preview_bbox=bbox,
        preview_face_crop_rgb=getattr(decision, "preview_face_crop_rgb", None),
        frame_previews=list(getattr(decision, "frame_previews", []) or []),
        elapsed_sec=elapsed,
        is_mock_preview=False,
    )


def run_ui_preview_simulation(
    frames_rgb: List[np.ndarray],
    config: Dict[str, Any],
) -> PipelineExecutionResult:
    """Chế độ kiểm thử giao diện (UI Dry-Run) khi nhánh src/ chưa merge đủ model.

    Chỉ được kích hoạt khi người dùng chọn rõ ràng chế độ Preview UI trong trường hợp
    `src.pipeline` chưa sẵn sàng; tuyệt đối không thay thế cho model thật.
    """
    start = time.perf_counter()
    max_frames = int(config.get("max_frames", DEFAULT_MAX_FRAMES))
    enable_deepfake = bool(config.get("enable_deepfake", True))
    sample_frames = frames_rgb[:max_frames] if frames_rgb else []

    if not sample_frames:
        return PipelineExecutionResult(
            final_label=LABEL_NO_FACE,
            confidence=0.0,
            aggregated_probs={
                LABEL_REAL: 0.0,
                LABEL_PRINTED: 0.0,
                LABEL_ONSCREEN: 0.0,
                LABEL_DEEPFAKE: 0.0,
            },
            reason="Không có frame hợp lệ trong chế độ xem trước UI.",
            elapsed_sec=time.perf_counter() - start,
            is_mock_preview=True,
        )

    first_frame = sample_frames[0]
    h, w = first_frame.shape[:2]
    bbox = (int(w * 0.25), int(h * 0.18), int(w * 0.75), int(h * 0.82))
    crop_rgb: Optional[np.ndarray] = None
    try:
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            pil_crop = Image.fromarray(first_frame.astype(np.uint8), mode="RGB").crop((x1, y1, x2, y2))
            crop_rgb = np.asarray(pil_crop.resize((224, 224)), dtype=np.uint8)
    except Exception:
        crop_rgb = None

    logs: List[dict] = []
    previews: List[dict] = []
    sample_fps = max(1, int(config.get("sample_fps", 5)))
    for idx, frm in enumerate(sample_frames):
        ts = round(idx / sample_fps, 2)
        logs.append(
            {
                "frame_id": idx,
                "timestamp_sec": ts,
                "has_face": True,
                "bbox": bbox,
                "frame_label": LABEL_REAL,
                "spoof_probs": {LABEL_REAL: 0.86, LABEL_PRINTED: 0.08, LABEL_ONSCREEN: 0.06},
                "fake_prob": 0.12 if enable_deepfake else None,
                "latency_ms": 18.5,
            }
        )
        previews.append(
            {
                "frame_id": idx,
                "timestamp_sec": ts,
                "has_face": True,
                "face_count": 1,
                "frame_label": LABEL_REAL,
                "frame_confidence": 0.86,
                "frame_rgb": frm,
                "faces": [
                    {
                        "face_index": 1,
                        "bbox": bbox,
                        "orig_bbox": bbox,
                        "det_confidence": 0.99,
                        "label": LABEL_REAL,
                        "confidence": 0.86,
                        "spoof_probs": {LABEL_REAL: 0.86, LABEL_PRINTED: 0.08, LABEL_ONSCREEN: 0.06},
                        "fake_prob": 0.12 if enable_deepfake else None,
                        "face_crop_rgb": crop_rgb,
                    }
                ],
                "spoof_probs": {LABEL_REAL: 0.86, LABEL_PRINTED: 0.08, LABEL_ONSCREEN: 0.06},
                "fake_prob": 0.12 if enable_deepfake else None,
                "latency_ms": 18.5,
            }
        )

    return PipelineExecutionResult(
        final_label=LABEL_REAL,
        confidence=0.86,
        aggregated_probs={
            LABEL_REAL: 0.86,
            LABEL_PRINTED: 0.08,
            LABEL_ONSCREEN: 0.06,
            LABEL_DEEPFAKE: 0.12 if enable_deepfake else 0.0,
        },
        reason="[UI Preview] Chế độ kiểm thử giao diện (chưa nối trọng số model trong src/).",
        frame_logs=logs,
        preview_frame_rgb=first_frame,
        preview_bbox=bbox,
        preview_face_crop_rgb=crop_rgb,
        frame_previews=previews,
        elapsed_sec=time.perf_counter() - start,
        is_mock_preview=True,
    )


def _encode_rgb_png_bytes(frame_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    arr = np.asarray(frame_rgb, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def run_backend_frames_pipeline(
    frames_rgb: Sequence[np.ndarray],
    config: Dict[str, Any],
) -> PipelineExecutionResult:
    """Chạy pipeline thật trên chuỗi khung hình RGB trong bộ đệm (ví dụ webcam trình duyệt)."""
    ensure_project_root_on_path()
    start = time.perf_counter()
    max_frames = max(1, int(config.get("max_frames", DEFAULT_MAX_FRAMES)))
    sample_fps = max(1, int(config.get("sample_fps", DEFAULT_SAMPLE_FPS)))
    enable_deepfake = bool(config.get("enable_deepfake", True))
    selected = list(frames_rgb)[-max_frames:] if frames_rgb else []

    if not selected:
        return PipelineExecutionResult(
            final_label=LABEL_NO_FACE,
            confidence=0.0,
            aggregated_probs={
                LABEL_REAL: 0.0,
                LABEL_PRINTED: 0.0,
                LABEL_ONSCREEN: 0.0,
                LABEL_DEEPFAKE: 0.0,
            },
            reason=UI_ERRORS["no_face"],
            elapsed_sec=time.perf_counter() - start,
            is_mock_preview=False,
        )

    if len(selected) == 1:
        png_bytes = _encode_rgb_png_bytes(selected[0])
        with temporary_memory_file(png_bytes, suffix=".png") as temp_path:
            return run_backend_pipeline(temp_path, config, fallback_frame_rgb=selected[0])

    try:
        pipeline_mod = importlib.import_module("src.pipeline")
        run_pipeline_fn = getattr(pipeline_mod, "run_pipeline")
        schemas_mod = importlib.import_module("src.schemas")
        SpoofScore = getattr(schemas_mod, "SpoofScore")
        DeepfakeScore = getattr(schemas_mod, "DeepfakeScore")
        aggregator_mod = importlib.import_module("src.decision.aggregator")
        aggregate_decision_fn = getattr(aggregator_mod, "aggregate_decision")
    except Exception as exc:
        return PipelineExecutionResult(
            final_label=LABEL_NO_FACE,
            confidence=0.0,
            aggregated_probs={
                LABEL_REAL: 0.0,
                LABEL_PRINTED: 0.0,
                LABEL_ONSCREEN: 0.0,
                LABEL_DEEPFAKE: 0.0,
            },
            reason=f"Không nạp được module pipeline ({type(exc).__name__}): {exc}",
            elapsed_sec=time.perf_counter() - start,
            error_message=f"Lỗi xử lý pipeline ({type(exc).__name__}): {exc}",
        )

    all_logs: List[dict] = []
    all_previews: List[dict] = []
    spoof_scores: List[Any] = []
    df_scores: List[Any] = []
    first_preview_rgb: Optional[np.ndarray] = selected[0]
    first_bbox: Optional[Tuple[int, int, int, int]] = None
    first_crop_rgb: Optional[np.ndarray] = None
    single_cfg = {**config, "max_frames": 1}

    try:
        for idx, frame in enumerate(selected):
            png_bytes = _encode_rgb_png_bytes(frame)
            with temporary_memory_file(png_bytes, suffix=".png") as temp_path:
                dec = run_pipeline_fn(temp_path, single_cfg)

            dec_logs = list(getattr(dec, "frame_logs", []) or [])
            if dec_logs:
                entry = dict(dec_logs[0])
            else:
                entry = {
                    "frame_id": idx,
                    "timestamp_sec": round(idx / sample_fps, 2),
                    "has_face": False,
                    "bbox": None,
                    "frame_label": LABEL_NO_FACE,
                    "spoof_probs": {LABEL_REAL: 0.0, LABEL_PRINTED: 0.0, LABEL_ONSCREEN: 0.0},
                    "fake_prob": None,
                    "latency_ms": 0.0,
                }
            entry["frame_id"] = idx
            entry["timestamp_sec"] = round(idx / sample_fps, 2)
            all_logs.append(entry)

            dec_previews = list(getattr(dec, "frame_previews", []) or [])
            if dec_previews:
                pv = dict(dec_previews[0])
                pv["frame_id"] = idx
                pv["timestamp_sec"] = round(idx / sample_fps, 2)
                all_previews.append(pv)

            if entry.get("has_face"):
                if first_bbox is None:
                    raw_preview = getattr(dec, "preview_frame_rgb", None)
                    first_preview_rgb = raw_preview if raw_preview is not None else frame
                    raw_bbox = getattr(dec, "preview_bbox", None)
                    first_bbox = raw_bbox if raw_bbox is not None else entry.get("bbox")
                    first_crop_rgb = getattr(dec, "preview_face_crop_rgb", None)

                s_probs = dict(entry.get("spoof_probs") or {})
                spoof_scores.append(
                    SpoofScore(
                        frame_id=idx,
                        probs={
                            LABEL_REAL: float(s_probs.get(LABEL_REAL, 0.0)),
                            LABEL_PRINTED: float(s_probs.get(LABEL_PRINTED, 0.0)),
                            LABEL_ONSCREEN: float(s_probs.get(LABEL_ONSCREEN, 0.0)),
                        },
                        latency_ms=float(entry.get("latency_ms", 0.0)),
                    )
                )
                if enable_deepfake and entry.get("fake_prob") is not None:
                    df_scores.append(
                        DeepfakeScore(
                            frame_id=idx,
                            fake_prob=float(entry["fake_prob"]),
                            latency_ms=float(entry.get("latency_ms", 0.0)),
                        )
                    )

        decision = aggregate_decision_fn(
            spoof_scores,
            df_scores if (enable_deepfake and df_scores) else None,
            config.get("thresholds"),
            frame_logs=all_logs,
            preview_frame_rgb=first_preview_rgb,
            preview_bbox=first_bbox,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        err_name = type(exc).__name__
        ui_msg = UI_ERRORS["invalid_file"] if err_name == "InvalidInputError" else f"Lỗi xử lý pipeline ({err_name}): {exc}"
        return PipelineExecutionResult(
            final_label=LABEL_NO_FACE,
            confidence=0.0,
            aggregated_probs={
                LABEL_REAL: 0.0,
                LABEL_PRINTED: 0.0,
                LABEL_ONSCREEN: 0.0,
                LABEL_DEEPFAKE: 0.0,
            },
            reason=ui_msg,
            elapsed_sec=elapsed,
            error_message=ui_msg,
        )

    elapsed = time.perf_counter() - start
    return PipelineExecutionResult(
        final_label=str(getattr(decision, "final_label", LABEL_NO_FACE)),
        confidence=float(getattr(decision, "confidence", 0.0)),
        aggregated_probs=dict(getattr(decision, "aggregated_probs", {}) or {}),
        reason=str(getattr(decision, "reason", "")),
        frame_logs=all_logs,
        preview_frame_rgb=first_preview_rgb,
        preview_bbox=first_bbox,
        preview_face_crop_rgb=first_crop_rgb,
        frame_previews=all_previews,
        elapsed_sec=elapsed,
        is_mock_preview=False,
    )
