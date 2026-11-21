"""Cầu nối giao tiếp giữa Streamlit UI (M6) và src/pipeline.py (M1)."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np

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
) -> Dict[str, Any]:
    """Tạo dict config đồng bộ cho `src.pipeline.run_pipeline(source, config)`."""
    return {
        "sample_fps": int(sample_fps),
        "max_frames": int(max_frames),
        "enable_deepfake": bool(enable_deepfake),
        "thresholds": {
            "tau_spoof": float(tau_spoof),
            "tau_deepfake": float(tau_deepfake),
            "uncertain_margin": float(uncertain_margin),
            # Alias dự phòng nếu M1 đọc key ngắn
            "spoof": float(tau_spoof),
            "deepfake": float(tau_deepfake),
        },
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
    source: Union[str, int],
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

    logs: List[dict] = []
    for idx, _ in enumerate(sample_frames):
        logs.append(
            {
                "frame_id": idx,
                "timestamp_sec": round(idx / max(1, int(config.get("sample_fps", 5))), 2),
                "has_face": True,
                "bbox": bbox,
                "frame_label": LABEL_REAL,
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
        elapsed_sec=time.perf_counter() - start,
        is_mock_preview=True,
    )
