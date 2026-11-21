"""Công cụ vẽ bounding box, định dạng dòng thời gian và cảnh báo ngưỡng cho M6."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from app.components.constants import (
    AGGREGATED_CLASSES,
    LABEL_BADGE_COLORS,
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


def draw_face_annotation(
    frame_rgb: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    label: str,
    confidence: float,
) -> np.ndarray:
    """Vẽ bounding box (x1, y1, x2, y2) và nhãn lên ảnh RGB uint8 (không đổi hệ màu)."""
    if frame_rgb is None or frame_rgb.ndim != 3:
        raise ValueError("frame_rgb phải là mảng numpy 3 chiều (H, W, 3) hệ RGB.")

    annotated = np.array(frame_rgb, dtype=np.uint8, copy=True)
    if bbox is None:
        return annotated

    h, w = annotated.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    color_rgb = _hex_to_rgb(LABEL_BADGE_COLORS.get(label, "#0f172a"))
    caption = f"{label} ({confidence * 100:.1f}%)"

    if cv2 is not None:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_rgb, 2)
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_y = max(y1 - 8, th + 6)
        cv2.rectangle(
            annotated,
            (x1, max(0, text_y - th - 6)),
            (min(w, x1 + tw + 10), text_y + 2),
            color_rgb,
            -1,
        )
        cv2.putText(
            annotated,
            caption,
            (x1 + 5, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    if Image is not None and ImageDraw is not None:
        pil_img = Image.fromarray(annotated, mode="RGB")
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)
        text_y = max(0, y1 - 18)
        draw.rectangle([x1, text_y, min(w, x1 + len(caption) * 7 + 8), text_y + 18], fill=color_rgb)
        draw.text((x1 + 4, text_y + 3), caption, fill=(255, 255, 255))
        return np.asarray(pil_img, dtype=np.uint8)

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
