"""End-to-end inference pipeline connecting M3 (Video I/O & Face Detection) to M6 Streamlit UI.

When M4 (`src.anti_spoof.predictor`), M5 (`src.deepfake.predictor`), and
M1 (`src.decision.aggregator`) are present, they are invoked automatically using
the frozen interface in `src/schemas.py`. When only M3 is present, the pipeline
executes real frame extraction and YuNet face detection/cropping and reports
transparent M3 diagnostics alongside neutral placeholder branch scores.
"""

import importlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.face_detection.detector import FaceDetector
from src.schemas import DeepfakeScore, FaceCropResult, FinalDecision, FramePacket, SpoofScore
from src.video_io.reader import load_frames


def _load_optional_callable(module_path: str, attr_name: str) -> Optional[Callable[..., Any]]:
    try:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, attr_name, None)
        return fn if callable(fn) else None
    except Exception:
        return None


_cached_detector: Optional[FaceDetector] = None


def _get_detector() -> FaceDetector:
    global _cached_detector
    if _cached_detector is None:
        _cached_detector = FaceDetector()
    return _cached_detector


def run_pipeline(source: Union[str, int], config: Optional[dict] = None) -> FinalDecision:
    """Chạy pipeline từ nguồn ảnh/video/webcam tới FinalDecision theo Quy ước Mục II.4."""
    cfg = config or {}
    sample_fps = int(cfg.get("sample_fps", 5))
    max_frames = int(cfg.get("max_frames", 30))
    enable_deepfake = bool(cfg.get("enable_deepfake", True))
    scale_ratio = float(cfg.get("scale_ratio", 1.2))
    thresholds = dict(cfg.get("thresholds", {}))

    packets: List[FramePacket] = load_frames(
        source=source,
        sample_fps=sample_fps,
        max_frames=max_frames,
    )

    detector = _get_detector()
    predict_spoof_fn = _load_optional_callable("src.anti_spoof.predictor", "predict_spoof")
    predict_deepfake_fn = (
        _load_optional_callable("src.deepfake.predictor", "predict_deepfake")
        if enable_deepfake
        else None
    )
    aggregate_fn = _load_optional_callable("src.decision.aggregator", "aggregate_decision")

    spoof_scores: List[SpoofScore] = []
    df_scores: List[DeepfakeScore] = []
    frame_logs: List[dict] = []
    valid_crops: List[FaceCropResult] = []

    preview_frame_rgb: Optional[np.ndarray] = None
    preview_bbox: Optional[Tuple[int, int, int, int]] = None
    preview_face_crop_rgb: Optional[np.ndarray] = None

    for pkt in packets:
        t0 = time.perf_counter()
        crop = detector.detect_and_crop(pkt, target_size=(224, 224), scale_ratio=scale_ratio)
        det_ms = (time.perf_counter() - t0) * 1000.0
        face_count = detector.last_face_count

        if preview_frame_rgb is None:
            preview_frame_rgb = pkt.image_rgb

        if not crop.has_face:
            frame_logs.append(
                {
                    "frame_id": pkt.frame_id,
                    "timestamp_sec": pkt.timestamp_sec,
                    "has_face": False,
                    "face_count": face_count,
                    "bbox": None,
                    "det_confidence": 0.0,
                    "frame_label": "No_Face",
                    "spoof_probs": {"Real": 0.0, "Printed": 0.0, "On-screen": 0.0},
                    "fake_prob": None,
                    "latency_ms": round(det_ms, 2),
                }
            )
            continue

        valid_crops.append(crop)
        if preview_bbox is None:
            preview_frame_rgb = pkt.image_rgb
            preview_bbox = crop.bbox
            preview_face_crop_rgb = crop.face_rgb

        branch_ms = 0.0
        if predict_spoof_fn is not None:
            spoof_score = predict_spoof_fn(crop)
            spoof_probs = dict(spoof_score.probs)
            branch_ms += float(getattr(spoof_score, "latency_ms", 0.0))
            spoof_scores.append(spoof_score)
        else:
            spoof_probs = {"Real": 1.0, "Printed": 0.0, "On-screen": 0.0}
            spoof_scores.append(SpoofScore(frame_id=pkt.frame_id, probs=spoof_probs, latency_ms=0.0))

        fake_prob: Optional[float] = None
        if enable_deepfake:
            if predict_deepfake_fn is not None:
                df_score = predict_deepfake_fn(crop)
                fake_prob = float(df_score.fake_prob)
                branch_ms += float(getattr(df_score, "latency_ms", 0.0))
                df_scores.append(df_score)
            else:
                fake_prob = 0.0
                df_scores.append(DeepfakeScore(frame_id=pkt.frame_id, fake_prob=0.0, latency_ms=0.0))

        frame_label = max(spoof_probs, key=lambda k: spoof_probs[k])
        if fake_prob is not None and fake_prob >= float(thresholds.get("tau_deepfake", 0.60)):
            frame_label = "Deepfake"

        frame_logs.append(
            {
                "frame_id": pkt.frame_id,
                "timestamp_sec": pkt.timestamp_sec,
                "has_face": True,
                "face_count": face_count,
                "bbox": crop.bbox,
                "det_confidence": crop.det_confidence,
                "frame_label": frame_label,
                "spoof_probs": spoof_probs,
                "fake_prob": fake_prob,
                "latency_ms": round(det_ms + branch_ms, 2),
            }
        )

    if aggregate_fn is not None and predict_spoof_fn is not None:
        decision = aggregate_fn(
            spoof_scores,
            df_scores if (enable_deepfake and predict_deepfake_fn is not None) else None,
            thresholds,
        )
        if not getattr(decision, "frame_logs", None):
            decision.frame_logs = frame_logs
    elif not valid_crops:
        decision = FinalDecision(
            final_label="No_Face",
            confidence=0.0,
            aggregated_probs={
                "Real": 0.0,
                "Printed": 0.0,
                "On-screen": 0.0,
                "Deepfake": 0.0,
            },
            reason=(
                f"M3 (Video I/O + YuNet) đã kiểm tra {len(packets)} frame "
                "nhưng không phát hiện khuôn mặt nào."
            ),
            frame_logs=frame_logs,
        )
    else:
        avg_det_conf = sum(c.det_confidence for c in valid_crops) / len(valid_crops)
        missing_branches: List[str] = []
        if predict_spoof_fn is None:
            missing_branches.append("M4 (Anti-Spoof)")
        if enable_deepfake and predict_deepfake_fn is None:
            missing_branches.append("M5 (Deepfake)")

        if missing_branches:
            missing_note = (
                f" Lưu ý: Nhánh {' & '.join(missing_branches)} chưa được tích hợp trọng số — "
                "hiện hiển thị kết quả trích xuất & nhận diện khuôn mặt thật từ M3."
            )
        else:
            missing_note = ""

        decision = FinalDecision(
            final_label="Real",
            confidence=round(avg_det_conf, 4),
            aggregated_probs={
                "Real": 1.0,
                "Printed": 0.0,
                "On-screen": 0.0,
                "Deepfake": 0.0,
            },
            reason=(
                f"M3 (Video I/O + YuNet) phát hiện khuôn mặt ở {len(valid_crops)}/{len(packets)} frame "
                f"(det_conf TB={avg_det_conf:.2f}, crop 224x224 RGB uint8).{missing_note}"
            ),
            frame_logs=frame_logs,
        )

    # Đính kèm khung hình, bbox và ảnh crop chuẩn M3 để UI Streamlit hiển thị trực tiếp
    setattr(decision, "preview_frame_rgb", preview_frame_rgb)
    setattr(decision, "preview_bbox", preview_bbox)
    setattr(decision, "preview_face_crop_rgb", preview_face_crop_rgb)
    return decision
