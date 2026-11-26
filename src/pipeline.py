"""End-to-end inference pipeline connecting M3 (Video I/O & Face Detection) to M6 Streamlit UI.

When M4 (`src.anti_spoof.predictor`), M5 (`src.deepfake.predictor`), and
M1 (`src.decision.aggregator`) are present, they are invoked automatically using
the frozen interface in `src/schemas.py`. When only M3 is present, the pipeline
executes real frame extraction and YuNet face detection/cropping and reports
transparent M3 diagnostics alongside neutral placeholder branch scores.
"""

from functools import lru_cache
import importlib
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import yaml

from src.face_detection.detector import FaceDetector, crop_face
from src.schemas import DeepfakeScore, FaceCropResult, FinalDecision, FramePacket, SpoofScore
from src.video_io.reader import load_frames

_DEFAULT_YAML_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@lru_cache(maxsize=1)
def load_default_yaml_config() -> Dict[str, Any]:
    """Đọc cấu hình mặc định từ `configs/default.yaml` (cache một lần)."""
    try:
        if not _DEFAULT_YAML_PATH.is_file():
            return {}
        data = yaml.safe_load(_DEFAULT_YAML_PATH.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _merge_pipeline_config(config: Optional[dict]) -> Dict[str, Any]:
    base = load_default_yaml_config()
    caller = dict(config or {})
    merged = {**base, **caller}
    base_thr = dict(base.get("thresholds") or {}) if isinstance(base.get("thresholds"), dict) else {}
    caller_thr = dict(caller.get("thresholds") or {}) if isinstance(caller.get("thresholds"), dict) else {}
    merged["thresholds"] = {**base_thr, **caller_thr}
    return merged


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


def _classify_face_verdict(
    spoof_probs: Dict[str, float],
    fake_prob: Optional[float],
    tau_spoof: float,
    tau_deepfake: float,
    enable_deepfake: bool,
) -> Tuple[str, float]:
    """Phân loại 1 khuôn mặt theo thứ tự ưu tiên DD6."""
    p_real = float(spoof_probs.get("Real", 0.0))
    p_printed = float(spoof_probs.get("Printed", 0.0))
    p_onscreen = float(spoof_probs.get("On-screen", 0.0))
    spoof_sum = p_printed + p_onscreen

    if spoof_sum > tau_spoof:
        if p_printed >= p_onscreen:
            return "Printed", round(max(p_printed, spoof_sum), 4)
        return "On-screen", round(max(p_onscreen, spoof_sum), 4)

    if enable_deepfake and fake_prob is not None and float(fake_prob) > tau_deepfake:
        return "Deepfake", round(float(fake_prob), 4)

    conf = min(p_real, 1.0 - float(fake_prob)) if (enable_deepfake and fake_prob is not None) else p_real
    return "Real", round(max(0.0, min(1.0, conf)), 4)


def _resize_scene_preview(image_rgb: np.ndarray, max_width: int = 960) -> Tuple[np.ndarray, float]:
    """Thu gọn ảnh cảnh xem trước để giữ nhẹ RAM nhưng vẫn sắc nét trên UI."""
    h, w = image_rgb.shape[:2]
    if w <= max_width or w <= 0:
        return image_rgb, 1.0
    scale = max_width / float(w)
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8), scale


def _select_representative_face_idx(
    faces_eval: List[Dict[str, Any]],
    strategy: str,
    tau_spoof: float,
    tau_deepfake: float,
    enable_deepfake: bool,
) -> int:
    """Chọn chỉ số khuôn mặt đại diện cho frame (`primary` = #1 lớn nhất, `highest_risk` = rủi ro cao nhất)."""
    if strategy != "highest_risk" or len(faces_eval) <= 1:
        return 0

    def _risk_key(item: Dict[str, Any]) -> Tuple[int, int, float]:
        s_probs = item["spoof_probs"]
        spoof_sum = float(s_probs.get("Printed", 0.0)) + float(s_probs.get("On-screen", 0.0))
        fake_p = item["fake_prob"]
        is_spoof = 1 if spoof_sum > tau_spoof else 0
        is_df = 1 if (enable_deepfake and fake_p is not None and float(fake_p) > tau_deepfake) else 0
        peak = max(spoof_sum, float(fake_p) if fake_p is not None else 0.0)
        return (is_spoof, is_df, peak)

    return max(range(len(faces_eval)), key=lambda idx: _risk_key(faces_eval[idx]))


def _dampen_blurry_background_spoof(
    s_score: SpoofScore,
    face_crop: FaceCropResult,
    is_secondary: bool,
    orig_img_shape: Tuple[int, int],
) -> SpoofScore:
    """Giảm cảnh báo giả mạo vật lý nhầm trên khuôn mặt xóa phông mờ ở hậu cảnh."""
    if face_crop.face_rgb is None or face_crop.bbox is None:
        return s_score
    img_w = max(1, int(orig_img_shape[1]))
    bw = max(1, int(face_crop.bbox[2] - face_crop.bbox[0]))
    if not is_secondary and bw >= 0.12 * img_w:
        return s_score

    probs = dict(s_score.probs)
    p_print = float(probs.get("Printed", 0.0))
    p_screen = float(probs.get("On-screen", 0.0))
    if (p_print + p_screen) <= 0.35:
        return s_score

    gray = cv2.cvtColor(face_crop.face_rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    if lap_var >= 22.0:
        return s_score

    blur_factor = max(0.15, lap_var / 22.0)
    p_print_adj = p_print * blur_factor
    p_screen_adj = p_screen * blur_factor
    p_real_adj = float(probs.get("Real", 0.0)) + (p_print - p_print_adj) + (p_screen - p_screen_adj)
    total = p_real_adj + p_print_adj + p_screen_adj
    if total <= 0.0:
        return s_score

    return SpoofScore(
        frame_id=s_score.frame_id,
        probs={
            "Real": p_real_adj / total,
            "Printed": p_print_adj / total,
            "On-screen": p_screen_adj / total,
        },
        latency_ms=s_score.latency_ms,
    )


def run_pipeline(source: Union[str, int], config: Optional[dict] = None) -> FinalDecision:
    """Chạy pipeline từ nguồn ảnh/video/webcam tới FinalDecision theo Quy ước Mục II.4."""
    cfg = _merge_pipeline_config(config)
    sample_fps = int(cfg.get("sample_fps", 5))
    max_frames = int(cfg.get("max_frames", 30))
    enable_deepfake = bool(cfg.get("enable_deepfake", True))
    scale_ratio = float(cfg.get("scale_ratio", 1.2))
    spoof_scale_ratio = float(cfg.get("spoof_scale_ratio", 2.7))
    sampling_mode = str(cfg.get("sampling_mode", "fps"))
    multi_face = bool(cfg.get("multi_face", False))
    max_faces_per_frame = int(cfg.get("max_faces_per_frame", 20))
    det_conf_threshold = float(cfg.get("det_conf_threshold", 0.62))
    include_frame_previews = bool(cfg.get("include_frame_previews", False))
    deepfake_variant = str(cfg.get("deepfake_variant", "siglip2"))
    aggregation_strategy = str(cfg.get("aggregation_strategy", "primary"))
    thresholds = dict(cfg.get("thresholds", {}))
    tau_spoof = float(thresholds.get("tau_spoof", thresholds.get("spoof", 0.60)))
    tau_deepfake = float(thresholds.get("tau_deepfake", thresholds.get("deepfake", 0.65)))

    if sampling_mode == "fps":
        packets: List[FramePacket] = load_frames(
            source=source,
            sample_fps=sample_fps,
            max_frames=max_frames,
        )
    else:
        packets = load_frames(
            source=source,
            sample_fps=sample_fps,
            max_frames=max_frames,
            sampling_mode=sampling_mode,
        )

    detector = _get_detector()
    predict_spoof_fn = _load_optional_callable("src.anti_spoof.predictor", "predict_spoof")
    predict_deepfake_fn = (
        _load_optional_callable("src.deepfake.predictor", "predict_deepfake")
        if enable_deepfake
        else None
    )
    predict_deepfake_batch_fn = (
        _load_optional_callable("src.deepfake.predictor", "predict_deepfake_batch")
        if enable_deepfake
        else None
    )
    aggregate_fn = _load_optional_callable("src.decision.aggregator", "aggregate_decision")

    preview_frame_rgb: Optional[np.ndarray] = None
    preview_bbox: Optional[Tuple[int, int, int, int]] = None
    preview_face_crop_rgb: Optional[np.ndarray] = None
    valid_crops: List[FaceCropResult] = []

    # Pass A: Phát hiện khuôn mặt + cắt kép (2.7x cho Anti-Spoof, 1.2x cho Deepfake) + chạy GĐ 1
    frame_records: List[Dict[str, Any]] = []
    flat_df_crops: List[FaceCropResult] = []
    flat_df_coords: List[Tuple[int, int]] = []  # (frame_record_idx, face_idx)

    for pkt in packets:
        t0 = time.perf_counter()
        if multi_face and hasattr(detector, "detect_all_and_crop"):
            crops = detector.detect_all_and_crop(
                pkt,
                target_size=(224, 224),
                scale_ratio=scale_ratio,
                max_faces=max_faces_per_frame,
                secondary_conf_threshold=det_conf_threshold,
            )
        else:
            single_crop = detector.detect_and_crop(pkt, target_size=(224, 224), scale_ratio=scale_ratio)
            crops = [single_crop] if single_crop.has_face else []
        det_ms = (time.perf_counter() - t0) * 1000.0
        face_count = detector.last_face_count

        if preview_frame_rgb is None:
            preview_frame_rgb = pkt.image_rgb

        if not crops:
            frame_records.append(
                {
                    "pkt": pkt,
                    "det_ms": det_ms,
                    "face_count": face_count,
                    "crops": [],
                    "spoof_results": [],
                    "spoof_ms": 0.0,
                    "df_results": [],
                }
            )
            continue

        primary_crop = crops[0]
        valid_crops.append(primary_crop)
        if preview_bbox is None:
            preview_frame_rgb = pkt.image_rgb
            preview_bbox = primary_crop.bbox
            preview_face_crop_rgb = primary_crop.face_rgb

        spoof_results: List[SpoofScore] = []
        spoof_ms = 0.0
        rec_idx = len(frame_records)

        for f_idx, c_item in enumerate(crops):
            # GĐ 1 (2.7_80x80_MiniFASNetV2) cần vùng ngữ cảnh rộng 2.7x để thấy viền giấy/màn hình
            if c_item.bbox is not None and abs(spoof_scale_ratio - scale_ratio) > 1e-3:
                wide_rgb = crop_face(
                    pkt.image_rgb,
                    c_item.bbox,
                    target_size=(224, 224),
                    scale_ratio=spoof_scale_ratio,
                )
                spoof_crop = (
                    FaceCropResult(
                        frame_id=c_item.frame_id,
                        has_face=True,
                        bbox=c_item.bbox,
                        det_confidence=c_item.det_confidence,
                        face_rgb=wide_rgb,
                        face_count=c_item.face_count,
                    )
                    if wide_rgb is not None
                    else c_item
                )
            else:
                spoof_crop = c_item

            if predict_spoof_fn is not None:
                s_score = predict_spoof_fn(spoof_crop)
                if multi_face:
                    s_score = _dampen_blurry_background_spoof(
                        s_score,
                        c_item,
                        is_secondary=(f_idx > 0),
                        orig_img_shape=pkt.image_rgb.shape[:2],
                    )
                spoof_ms += float(getattr(s_score, "latency_ms", 0.0))
            else:
                s_score = SpoofScore(
                    frame_id=pkt.frame_id,
                    probs={"Real": 1.0, "Printed": 0.0, "On-screen": 0.0},
                    latency_ms=0.0,
                )
            spoof_results.append(s_score)
            flat_df_crops.append(c_item)
            flat_df_coords.append((rec_idx, f_idx))

        frame_records.append(
            {
                "pkt": pkt,
                "det_ms": det_ms,
                "face_count": face_count,
                "crops": crops,
                "spoof_results": spoof_results,
                "spoof_ms": spoof_ms,
                "df_results": [None] * len(crops),
            }
        )

    # Pass B: Suy luận Deepfake (GĐ 2) — dùng batching chunk=8 nếu không bị monkeypatch trong test
    if enable_deepfake and flat_df_crops:
        is_canonical_df = (
            predict_deepfake_fn is not None
            and getattr(predict_deepfake_fn, "__module__", "") == "src.deepfake.predictor"
            and getattr(predict_deepfake_fn, "__name__", "") == "predict_deepfake"
        )
        if is_canonical_df and predict_deepfake_batch_fn is not None:
            chunk_size = 8
            batched_scores: List[DeepfakeScore] = []
            for start_i in range(0, len(flat_df_crops), chunk_size):
                chunk = flat_df_crops[start_i : start_i + chunk_size]
                try:
                    batched_scores.extend(predict_deepfake_batch_fn(chunk, variant=deepfake_variant))
                except TypeError:
                    batched_scores.extend(predict_deepfake_batch_fn(chunk))
            for (r_idx, f_idx), d_score in zip(flat_df_coords, batched_scores):
                frame_records[r_idx]["df_results"][f_idx] = d_score
        else:
            for (r_idx, f_idx), c_item in zip(flat_df_coords, flat_df_crops):
                if predict_deepfake_fn is not None:
                    d_score = predict_deepfake_fn(c_item)
                else:
                    d_score = DeepfakeScore(frame_id=c_item.frame_id, fake_prob=0.0, latency_ms=0.0)
                frame_records[r_idx]["df_results"][f_idx] = d_score

    # Pass C: Tổng hợp log, preview từng cảnh & điểm đầu vào cho M1
    spoof_scores: List[SpoofScore] = []
    df_scores: List[DeepfakeScore] = []
    frame_logs: List[dict] = []
    frame_previews: List[dict] = []

    for rec in frame_records:
        pkt: FramePacket = rec["pkt"]
        crops: List[FaceCropResult] = rec["crops"]
        det_ms = float(rec["det_ms"])
        face_count = int(rec["face_count"])

        if not crops:
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
                    "faces": [],
                }
            )
            if include_frame_previews:
                scaled_rgb, _scale = _resize_scene_preview(pkt.image_rgb)
                frame_previews.append(
                    {
                        "frame_id": pkt.frame_id,
                        "timestamp_sec": pkt.timestamp_sec,
                        "has_face": False,
                        "face_count": face_count,
                        "frame_label": "No_Face",
                        "frame_confidence": 0.0,
                        "frame_rgb": scaled_rgb,
                        "faces": [],
                        "spoof_probs": {"Real": 0.0, "Printed": 0.0, "On-screen": 0.0},
                        "fake_prob": None,
                        "latency_ms": round(det_ms, 2),
                    }
                )
            continue

        faces_eval: List[Dict[str, Any]] = []
        faces_log_list: List[dict] = []
        faces_preview_raw: List[dict] = []
        df_ms = 0.0

        for f_idx, c_item in enumerate(crops):
            s_score: SpoofScore = rec["spoof_results"][f_idx]
            c_spoof_probs = dict(s_score.probs)
            d_score: Optional[DeepfakeScore] = rec["df_results"][f_idx]
            c_fake_prob: Optional[float] = None
            if enable_deepfake and d_score is not None:
                c_fake_prob = float(d_score.fake_prob)
                df_ms += float(getattr(d_score, "latency_ms", 0.0))

            f_label, f_conf = _classify_face_verdict(
                c_spoof_probs,
                c_fake_prob,
                tau_spoof=tau_spoof,
                tau_deepfake=tau_deepfake,
                enable_deepfake=enable_deepfake,
            )
            faces_eval.append(
                {
                    "crop": c_item,
                    "spoof_score": s_score,
                    "spoof_probs": c_spoof_probs,
                    "df_score": d_score,
                    "fake_prob": c_fake_prob,
                    "label": f_label,
                    "confidence": f_conf,
                }
            )
            faces_log_list.append(
                {
                    "face_index": f_idx + 1,
                    "bbox": list(c_item.bbox) if c_item.bbox is not None else None,
                    "det_confidence": c_item.det_confidence,
                    "label": f_label,
                    "confidence": f_conf,
                    "spoof_probs": c_spoof_probs,
                    "fake_prob": c_fake_prob,
                }
            )
            faces_preview_raw.append(
                {
                    "face_index": f_idx + 1,
                    "orig_bbox": c_item.bbox,
                    "det_confidence": c_item.det_confidence,
                    "label": f_label,
                    "confidence": f_conf,
                    "spoof_probs": c_spoof_probs,
                    "fake_prob": c_fake_prob,
                    "face_crop_rgb": c_item.face_rgb,
                }
            )

        rep_idx = _select_representative_face_idx(
            faces_eval,
            strategy=aggregation_strategy if multi_face else "primary",
            tau_spoof=tau_spoof,
            tau_deepfake=tau_deepfake,
            enable_deepfake=enable_deepfake,
        )
        rep = faces_eval[rep_idx]
        spoof_scores.append(rep["spoof_score"])
        if enable_deepfake and rep["df_score"] is not None:
            df_scores.append(rep["df_score"])

        primary_crop = crops[0]
        rep_spoof_probs = rep["spoof_probs"]
        rep_fake_prob = rep["fake_prob"]

        frame_label = faces_log_list[0]["label"]
        frame_confidence = faces_log_list[0]["confidence"]
        if multi_face:
            priority_rank = {"Printed": 3, "On-screen": 3, "Deepfake": 2, "Real": 1}
            top_face = max(
                faces_log_list,
                key=lambda item: (priority_rank.get(item["label"], 0), float(item["confidence"])),
            )
            frame_label = str(top_face["label"])
            frame_confidence = float(top_face["confidence"])

        total_latency_ms = round(det_ms + float(rec["spoof_ms"]) + df_ms, 2)
        frame_logs.append(
            {
                "frame_id": pkt.frame_id,
                "timestamp_sec": pkt.timestamp_sec,
                "has_face": True,
                "face_count": face_count,
                "bbox": primary_crop.bbox,
                "det_confidence": primary_crop.det_confidence,
                "frame_label": frame_label,
                "spoof_probs": rep_spoof_probs,
                "fake_prob": rep_fake_prob,
                "latency_ms": total_latency_ms,
                "faces": faces_log_list,
            }
        )

        if include_frame_previews:
            scaled_rgb, scale = _resize_scene_preview(pkt.image_rgb)
            scaled_faces: List[dict] = []
            for fp in faces_preview_raw:
                ob = fp["orig_bbox"]
                if ob is not None:
                    sb = (
                        int(round(ob[0] * scale)),
                        int(round(ob[1] * scale)),
                        int(round(ob[2] * scale)),
                        int(round(ob[3] * scale)),
                    )
                else:
                    sb = None
                scaled_faces.append(
                    {
                        "face_index": fp["face_index"],
                        "bbox": sb,
                        "orig_bbox": ob,
                        "det_confidence": fp["det_confidence"],
                        "label": fp["label"],
                        "confidence": fp["confidence"],
                        "spoof_probs": fp["spoof_probs"],
                        "fake_prob": fp["fake_prob"],
                        "face_crop_rgb": fp["face_crop_rgb"],
                    }
                )
            frame_previews.append(
                {
                    "frame_id": pkt.frame_id,
                    "timestamp_sec": pkt.timestamp_sec,
                    "has_face": True,
                    "face_count": face_count,
                    "frame_label": frame_label,
                    "frame_confidence": frame_confidence,
                    "frame_rgb": scaled_rgb,
                    "faces": scaled_faces,
                    "spoof_probs": rep_spoof_probs,
                    "fake_prob": rep_fake_prob,
                    "latency_ms": total_latency_ms,
                }
            )

    if aggregate_fn is not None and predict_spoof_fn is not None:
        decision = aggregate_fn(
            spoof_scores,
            df_scores if (enable_deepfake and predict_deepfake_fn is not None) else None,
            thresholds,
            frame_logs=frame_logs,
            preview_frame_rgb=preview_frame_rgb,
            preview_bbox=preview_bbox,
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

    # Đính kèm khung hình, bbox, crop và danh sách preview từng cảnh cho UI Streamlit
    setattr(decision, "preview_frame_rgb", preview_frame_rgb)
    setattr(decision, "preview_bbox", preview_bbox)
    setattr(decision, "preview_face_crop_rgb", preview_face_crop_rgb)
    setattr(decision, "frame_previews", frame_previews)
    return decision
