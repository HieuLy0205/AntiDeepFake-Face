"""Công cụ vẽ bounding box, định dạng dòng thời gian và cảnh báo ngưỡng cho M6."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from app.components.constants import (
    AGGREGATED_CLASSES,
    LABEL_BADGE_COLORS,
    LABEL_DISPLAY_VI,
    LABEL_NO_FACE,
    LABEL_ONSCREEN,
    LABEL_PRINTED,
    LABEL_REAL,
    LABEL_DEEPFAKE,
    UI_ERRORS,
)

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    cleaned = hex_color.lstrip("#")
    return tuple(int(cleaned[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore


_HUD_RGB_COLORS: Dict[str, Tuple[int, int, int]] = {
    LABEL_REAL: (16, 185, 129),
    LABEL_PRINTED: (245, 158, 11),
    LABEL_ONSCREEN: (249, 115, 22),
    LABEL_DEEPFAKE: (239, 68, 68),
    LABEL_NO_FACE: (148, 163, 184),
}


def _draw_single_hud_box(
    annotated: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    caption: str,
) -> None:
    """Vẽ một khung biometric HUD + nhãn lên ảnh `annotated` tại chỗ."""
    h, w = annotated.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    color_rgb = _HUD_RGB_COLORS.get(label) or _hex_to_rgb(LABEL_BADGE_COLORS.get(label, "#0f172a"))

    if cv2 is not None:
        bw = x2 - x1
        bh = y2 - y1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_rgb, 1, cv2.LINE_AA)
        corner_len = max(6, min(bw, bh) // 4)
        corner_len_x = min(corner_len, max(4, bw // 2))
        corner_len_y = min(corner_len, max(4, bh // 2))
        thickness = 1 if bw < 55 else max(2, min(w, h) // 180)

        cv2.line(annotated, (x1, y1), (x1 + corner_len_x, y1), color_rgb, thickness, cv2.LINE_AA)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len_y), color_rgb, thickness, cv2.LINE_AA)
        cv2.line(annotated, (x2, y1), (x2 - corner_len_x, y1), color_rgb, thickness, cv2.LINE_AA)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len_y), color_rgb, thickness, cv2.LINE_AA)
        cv2.line(annotated, (x1, y2), (x1 + corner_len_x, y2), color_rgb, thickness, cv2.LINE_AA)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_len_y), color_rgb, thickness, cv2.LINE_AA)
        cv2.line(annotated, (x2, y2), (x2 - corner_len_x, y2), color_rgb, thickness, cv2.LINE_AA)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_len_y), color_rgb, thickness, cv2.LINE_AA)

        font_scale = max(0.35, min(0.60, bw / 190.0))
        text_thick = 1
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)
        pad_x = 4 if bw < 65 else 7
        pad_y = 3 if bw < 65 else 5
        bar_w = 3 if bw < 65 else 5
        banner_h = th + pad_y * 2 + baseline
        banner_w = min(w - x1, tw + pad_x * 2 + bar_w)
        if y1 >= banner_h + 4:
            by1 = y1 - banner_h - 4
            by2 = y1 - 4
        else:
            by1 = min(h - banner_h, y1 + 2)
            by2 = min(h, by1 + banner_h)
        bx1 = x1
        bx2 = min(w, x1 + max(banner_w, bar_w + 12))

        if bx2 > bx1 and by2 > by1:
            roi = annotated[by1:by2, bx1:bx2]
            dark_bg = np.full_like(roi, (11, 17, 32), dtype=np.uint8)
            blended = cv2.addWeighted(dark_bg, 0.78, roi, 0.22, 0.0)
            annotated[by1:by2, bx1:bx2] = blended
            cv2.rectangle(annotated, (bx1, by1), (min(bx2, bx1 + bar_w), by2), color_rgb, -1)
            cv2.putText(
                annotated,
                caption,
                (bx1 + bar_w + pad_x, by2 - pad_y - baseline // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (248, 250, 252),
                text_thick,
                cv2.LINE_AA,
            )
        return

    if Image is not None and ImageDraw is not None:
        pil_img = Image.fromarray(annotated, mode="RGB")
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)
        text_y = max(0, y1 - 18)
        draw.rectangle([x1, text_y, min(w, x1 + len(caption) * 7 + 8), text_y + 18], fill=color_rgb)
        draw.text((x1 + 4, text_y + 3), caption, fill=(255, 255, 255))
        annotated[:] = np.asarray(pil_img, dtype=np.uint8)


def draw_face_annotation(
    frame_rgb: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    label: str,
    confidence: float,
) -> np.ndarray:
    """Vẽ bounding box kiểu biometric HUD (x1, y1, x2, y2) và nhãn lên ảnh RGB uint8."""
    if frame_rgb is None or frame_rgb.ndim != 3:
        raise ValueError("frame_rgb phải là mảng numpy 3 chiều (H, W, 3) hệ RGB.")

    annotated = np.array(frame_rgb, dtype=np.uint8, copy=True)
    if bbox is None:
        return annotated

    _draw_single_hud_box(annotated, bbox, label, f"{label} | {confidence * 100:.1f}%")
    return annotated


def draw_multi_face_annotations(
    frame_rgb: np.ndarray,
    faces: Sequence[Dict[str, Any]],
    fallback_bbox: Optional[Tuple[int, int, int, int]] = None,
    fallback_label: str = LABEL_REAL,
    fallback_confidence: float = 0.0,
) -> np.ndarray:
    """Vẽ bounding box HUD và nhãn cho toàn bộ khuôn mặt xuất hiện trong một khung cảnh."""
    if frame_rgb is None or frame_rgb.ndim != 3:
        raise ValueError("frame_rgb phải là mảng numpy 3 chiều (H, W, 3) hệ RGB.")

    annotated = np.array(frame_rgb, dtype=np.uint8, copy=True)
    valid_faces = [f for f in faces if isinstance(f, dict) and f.get("bbox") is not None]
    if not valid_faces:
        if fallback_bbox is not None:
            _draw_single_hud_box(
                annotated,
                fallback_bbox,
                fallback_label,
                f"{fallback_label} | {fallback_confidence * 100:.1f}%",
            )
        return annotated

    multi = len(valid_faces) > 1
    for idx, face in enumerate(valid_faces, start=1):
        bbox = tuple(int(v) for v in face["bbox"])
        lbl = str(face.get("label", LABEL_REAL))
        conf = float(face.get("confidence", 0.0))
        face_num = int(face.get("face_index", idx))
        bw = bbox[2] - bbox[0]
        if multi and bw < 70:
            caption = f"#{face_num} {lbl} {conf * 100:.0f}%"
        elif multi:
            caption = f"#{face_num} {lbl} | {conf * 100:.1f}%"
        else:
            caption = f"{lbl} | {conf * 100:.1f}%"
        _draw_single_hud_box(annotated, bbox, lbl, caption)  # type: ignore[arg-type]
    return annotated


def is_near_threshold(
    final_label: str,
    aggregated_probs: Dict[str, float],
    tau_spoof: float,
    tau_deepfake: float,
    margin: float,
) -> bool:
    """Kiểm tra điểm sát ngưỡng (Mục II.6 / DD6) để bật cảnh báo chưa chắc chắn."""
    if final_label == LABEL_NO_FACE or margin <= 0:
        return False

    spoof_score = float(aggregated_probs.get(LABEL_PRINTED, 0.0)) + float(
        aggregated_probs.get(LABEL_ONSCREEN, 0.0)
    )
    deepfake_score = float(aggregated_probs.get(LABEL_DEEPFAKE, 0.0))

    if abs(spoof_score - tau_spoof) <= margin:
        return True
    if final_label in (LABEL_REAL, LABEL_DEEPFAKE) and abs(deepfake_score - tau_deepfake) <= margin:
        return True
    return False


def has_multiple_faces_warning(frame_logs: Sequence[dict]) -> bool:
    """Kiểm tra xem có frame nào phát hiện >= 2 khuôn mặt hay không."""
    for entry in frame_logs:
        if not isinstance(entry, dict):
            continue
        face_count = entry.get("face_count") or entry.get("num_faces") or 0
        if isinstance(face_count, (int, float)) and face_count > 1:
            return True
        if entry.get("multi_face") is True:
            return True
    return False


def collect_decision_warnings(
    final_label: str,
    aggregated_probs: Dict[str, float],
    frame_logs: Sequence[dict],
    tau_spoof: float,
    tau_deepfake: float,
    uncertain_margin: float,
) -> List[str]:
    """Tổng hợp các cảnh báo chuẩn hóa theo Mục II.6 từ FinalDecision."""
    warnings: List[str] = []
    if final_label == LABEL_NO_FACE:
        warnings.append(UI_ERRORS["no_face"])
        return warnings

    if has_multiple_faces_warning(frame_logs):
        warnings.append(UI_ERRORS["multi_face"])

    if is_near_threshold(
        final_label=final_label,
        aggregated_probs=aggregated_probs,
        tau_spoof=tau_spoof,
        tau_deepfake=tau_deepfake,
        margin=uncertain_margin,
    ):
        warnings.append(UI_ERRORS["uncertain_score"])

    return warnings


def normalize_aggregated_probs(raw_probs: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Đảm bảo đủ 4 khóa chuẩn hóa và giá trị nằm trong [0.0, 1.0]."""
    probs = {cls_name: 0.0 for cls_name in AGGREGATED_CLASSES}
    if not raw_probs:
        return probs
    for cls_name in AGGREGATED_CLASSES:
        val = float(raw_probs.get(cls_name, 0.0))
        probs[cls_name] = max(0.0, min(1.0, val))
    return probs


def build_frame_timeline_rows(frame_logs: Sequence[dict]) -> List[Dict[str, Any]]:
    """Chuẩn hóa log từng frame từ M1 thành bảng hiển thị dòng thời gian trên UI."""
    rows: List[Dict[str, Any]] = []
    for idx, log in enumerate(frame_logs):
        if not isinstance(log, dict):
            continue
        frame_id = log.get("frame_id", idx)
        timestamp_sec = log.get("timestamp_sec")
        label = log.get("frame_label") or log.get("label") or LABEL_REAL

        spoof_probs = log.get("spoof_probs") or {}
        real_p = float(log.get("real_prob", spoof_probs.get(LABEL_REAL, 0.0)))
        printed_p = float(log.get("printed_prob", spoof_probs.get(LABEL_PRINTED, 0.0)))
        onscreen_p = float(log.get("onscreen_prob", spoof_probs.get(LABEL_ONSCREEN, 0.0)))
        fake_p = log.get("fake_prob")
        latency_ms = float(log.get("latency_ms", 0.0))

        rows.append(
            {
                "frame_id": int(frame_id),
                "timestamp_sec": round(float(timestamp_sec), 2)
                if timestamp_sec is not None
                else None,
                "label": str(label),
                "Real": round(real_p, 4),
                "Printed": round(printed_p, 4),
                "On-screen": round(onscreen_p, 4),
                "Deepfake": round(float(fake_p), 4) if fake_p is not None else None,
                "latency_ms": round(latency_ms, 2),
                "has_face": bool(log.get("has_face", label != LABEL_NO_FACE)),
                "face_count": int(log.get("face_count", 1 if log.get("has_face", label != LABEL_NO_FACE) else 0)),
            }
        )
    return rows


def compute_performance_stats(
    frame_logs: Sequence[dict],
    total_elapsed_sec: float,
) -> Dict[str, float]:
    """Tính FPS thực tế và độ trễ trung bình (phục vụ E6 và hiển thị UI)."""
    num_frames = len(frame_logs)
    effective_fps = (num_frames / total_elapsed_sec) if total_elapsed_sec > 0 and num_frames > 0 else 0.0
    latencies = [
        float(item["latency_ms"])
        for item in frame_logs
        if isinstance(item, dict) and item.get("latency_ms") is not None
    ]
    avg_latency_ms = (sum(latencies) / len(latencies)) if latencies else (
        (total_elapsed_sec * 1000.0 / num_frames) if num_frames > 0 else 0.0
    )
    return {
        "num_frames": float(num_frames),
        "total_elapsed_sec": round(total_elapsed_sec, 3),
        "effective_fps": round(effective_fps, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
    }


def build_export_report_dict(
    result: Any,
    config: Dict[str, Any],
    input_mode: str,
    source_name: str = "session_input",
) -> Dict[str, Any]:
    """Đóng gói kết quả phân tích thành dict thuần JSON (không kèm dữ liệu ảnh/video theo DD8)."""
    thresholds = dict(config.get("thresholds", {}) or {})
    tau_spoof = float(thresholds.get("tau_spoof", 0.60))
    tau_deepfake = float(thresholds.get("tau_deepfake", 0.65))
    uncertain_margin = float(thresholds.get("uncertain_margin", 0.08))

    final_label = str(getattr(result, "final_label", LABEL_NO_FACE))
    frame_logs = list(getattr(result, "frame_logs", []) or [])
    elapsed_sec = float(getattr(result, "elapsed_sec", 0.0))
    probs = normalize_aggregated_probs(getattr(result, "aggregated_probs", None))

    warnings = collect_decision_warnings(
        final_label=final_label,
        aggregated_probs=probs,
        frame_logs=frame_logs,
        tau_spoof=tau_spoof,
        tau_deepfake=tau_deepfake,
        uncertain_margin=uncertain_margin,
    )
    perf = compute_performance_stats(frame_logs, elapsed_sec)
    timeline = build_frame_timeline_rows(frame_logs)
    bbox = getattr(result, "preview_bbox", None)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_mode": str(input_mode),
        "source_name": str(source_name),
        "is_mock_preview": bool(getattr(result, "is_mock_preview", False)),
        "final_label": final_label,
        "final_label_vi": LABEL_DISPLAY_VI.get(final_label, final_label),
        "confidence": round(float(getattr(result, "confidence", 0.0)), 4),
        "reason": str(getattr(result, "reason", "")),
        "preview_bbox": [int(v) for v in bbox] if bbox is not None else None,
        "aggregated_probs": probs,
        "spoof_total_prob": round(probs.get(LABEL_PRINTED, 0.0) + probs.get(LABEL_ONSCREEN, 0.0), 4),
        "thresholds": {
            "tau_spoof": tau_spoof,
            "tau_deepfake": tau_deepfake,
            "uncertain_margin": uncertain_margin,
        },
        "sampling": {
            "sample_fps": int(config.get("sample_fps", 5)),
            "max_frames": int(config.get("max_frames", 10)),
            "enable_deepfake": bool(config.get("enable_deepfake", True)),
        },
        "performance": perf,
        "warnings": warnings,
        "frame_timeline": timeline,
    }
