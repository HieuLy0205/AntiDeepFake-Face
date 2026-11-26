"""End-to-end demo readiness audit & real-video M5 verification on CPU.

Audits every stage of the demo flow:
  Input -> Frame Extraction -> Face Detection/Crop (224x224 RGB)
        -> Stage 1 Anti-Spoof (M4) -> Stage 2 Deepfake (M5 - REAL checkpoints)
        -> Decision (DD6/DD7) -> UI Rendering (bbox, label, probs, timeline, FPS)

Also verifies all 3 trained M5 checkpoints on unseen `test` split MP4 videos (<= 30s)
using M6's actual `app/components/validators.py` and `app/components/visualizer.py`.
"""

from __future__ import annotations

import csv
import importlib
import json
import py_compile
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.components.constants import LABEL_DEEPFAKE, LABEL_NO_FACE, LABEL_ONSCREEN, LABEL_PRINTED, LABEL_REAL  # noqa: E402
from app.components.validators import validate_video_input  # noqa: E402
from app.components.visualizer import (  # noqa: E402
    build_frame_timeline_rows,
    collect_decision_warnings,
    compute_performance_stats,
    draw_face_annotation,
)
from src.deepfake import (  # noqa: E402
    DeepfakeDetector,
    EnsembleDeepfakeDetector,
    LightweightCNNDetector,
)
from src.schemas import DeepfakeScore, FaceCropResult, FinalDecision, FramePacket, SpoofScore  # noqa: E402


def audit_repo_stages() -> Dict[str, Dict[str, Any]]:
    """Audit real implementation status of every module in the demo pipeline."""
    report: Dict[str, Dict[str, Any]] = {}

    def _check_mod(name: str, attr: str) -> Dict[str, Any]:
        try:
            mod = importlib.import_module(name)
            has_attr = hasattr(mod, attr)
            return {"ready": has_attr, "detail": f"{name}.{attr} OK" if has_attr else f"Missing {name}.{attr}"}
        except Exception as exc:
            return {"ready": False, "detail": f"{type(exc).__name__}: {exc}"}

    report["M3_video_io"] = _check_mod("src.video_io.reader", "load_frames")
    report["M3_face_detection"] = _check_mod("src.face_detection.detector", "detect_and_crop")
    report["M4_anti_spoof"] = _check_mod("src.anti_spoof.predictor", "predict_spoof")
    report["M5_deepfake"] = _check_mod("src.deepfake.predictor", "predict_deepfake")
    report["M1_decision"] = _check_mod("src.decision.aggregator", "aggregate_decision")
    report["M1_pipeline"] = _check_mod("src.pipeline", "run_pipeline")

    try:
        py_compile.compile(str(REPO_ROOT / "app" / "app.py"), doraise=True)
        report["M6_streamlit_app"] = {"ready": True, "detail": "app/app.py syntax OK"}
    except py_compile.PyCompileError as exc:
        report["M6_streamlit_app"] = {"ready": False, "detail": str(exc.msg)}

    return report


def extract_frames_demo(video_path: Path, sample_fps: int = 5, max_frames: int = 10) -> List[FramePacket]:
    """Reference M3 video_io contract: sample <= max_frames RGB uint8 FramePackets."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    step = max(1, int(round(src_fps / max(1, sample_fps))))
    packets: List[FramePacket] = []
    raw_idx = 0
    try:
        while len(packets) < max_frames:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            if raw_idx % step == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                packets.append(
                    FramePacket(
                        frame_id=len(packets),
                        timestamp_sec=round(raw_idx / src_fps, 2),
                        image_rgb=np.ascontiguousarray(rgb, dtype=np.uint8),
                    )
                )
            raw_idx += 1
    finally:
        cap.release()
    return packets


def detect_largest_face_demo(mtcnn: Any, pkt: FramePacket, margin_ratio: float = 0.25) -> FaceCropResult:
    """Reference M3 face_detection contract: largest face -> (224, 224, 3) RGB uint8."""
    rgb = pkt.image_rgb
    h, w = rgb.shape[:2]
    scale = 640.0 / max(h, w) if max(h, w) > 640 else 1.0
    small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb

    boxes, probs = mtcnn.detect(Image.fromarray(small, mode="RGB"))
    if boxes is None or len(boxes) == 0:
        return FaceCropResult(frame_id=pkt.frame_id, has_face=False, face_count=0)

    valid_pairs = [(b, float(p)) for b, p in zip(boxes, probs) if p is not None and float(p) >= 0.80]
    if not valid_pairs:
        valid_pairs = [(boxes[0], float(probs[0] or 0.0))]

    best_box, best_prob = max(
        valid_pairs,
        key=lambda bp: max(0.0, float(bp[0][2] - bp[0][0])) * max(0.0, float(bp[0][3] - bp[0][1])),
    )
    x1 = int(round(float(best_box[0]) / scale))
    y1 = int(round(float(best_box[1]) / scale))
    x2 = int(round(float(best_box[2]) / scale))
    y2 = int(round(float(best_box[3]) / scale))

    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin_ratio)
    nx1 = max(0, int(round(cx - side / 2.0)))
    ny1 = max(0, int(round(cy - side / 2.0)))
    nx2 = min(w, int(round(cx + side / 2.0)))
    ny2 = min(h, int(round(cy + side / 2.0)))
    crop = rgb[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return FaceCropResult(frame_id=pkt.frame_id, has_face=False, face_count=0)

    face_224 = np.ascontiguousarray(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR), dtype=np.uint8)
    return FaceCropResult(
        frame_id=pkt.frame_id,
        has_face=True,
        bbox=(nx1, ny1, nx2, ny2),
        det_confidence=best_prob,
        face_rgb=face_224,
        face_count=len(valid_pairs),
    )


def aggregate_dd6_demo(
    spoof_scores: List[SpoofScore],
    df_scores: List[DeepfakeScore],
    tau_spoof: float = 0.60,
    tau_deepfake: float = 0.6693,
) -> Tuple[str, float, Dict[str, float], str]:
    """Reference M1 DD6/DD7 rule: No_Face -> Spoof (>tau_s) -> Deepfake (>tau_d) -> Real."""
    if not spoof_scores:
        return (
            LABEL_NO_FACE,
            0.0,
            {LABEL_REAL: 0.0, LABEL_PRINTED: 0.0, LABEL_ONSCREEN: 0.0, LABEL_DEEPFAKE: 0.0},
            "Không phát hiện khuôn mặt.",
        )
    mean_real = float(np.mean([s.probs[LABEL_REAL] for s in spoof_scores]))
    mean_printed = float(np.mean([s.probs[LABEL_PRINTED] for s in spoof_scores]))
    mean_onscreen = float(np.mean([s.probs[LABEL_ONSCREEN] for s in spoof_scores]))
    mean_fake = float(np.mean([d.fake_prob for d in df_scores])) if df_scores else 0.0
    probs = {
        LABEL_REAL: round(mean_real, 4),
        LABEL_PRINTED: round(mean_printed, 4),
        LABEL_ONSCREEN: round(mean_onscreen, 4),
        LABEL_DEEPFAKE: round(mean_fake, 4),
    }

    spoof_total = mean_printed + mean_onscreen
    if spoof_total > tau_spoof:
        lbl = LABEL_PRINTED if mean_printed >= mean_onscreen else LABEL_ONSCREEN
        return lbl, round(probs[lbl], 4), probs, f"Spoof={spoof_total:.3f} > tau_s={tau_spoof:.2f}"
    if df_scores and mean_fake >= tau_deepfake:
        return LABEL_DEEPFAKE, round(mean_fake, 4), probs, f"Deepfake={mean_fake:.3f} >= tau_d={tau_deepfake:.3f}"
    return LABEL_REAL, round(max(0.0, 1.0 - mean_fake), 4), probs, f"Real (spoof={spoof_total:.3f}, df={mean_fake:.3f})"


def select_demo_test_videos(metadata_csv: Path) -> List[dict]:
    """Pick 5 representative unseen `test` videos (<= 30s) covering Real + all Fake variants."""
    with metadata_csv.open("r", encoding="utf-8") as f:
        test_rows = [
            r for r in csv.DictReader(f)
            if r["split"] == "test" and 8.0 <= float(r.get("duration_sec") or 0.0) <= 25.0
        ]

    wanted = [
        ("Real (original)", lambda r: r["generator"] == "original"),
        ("InSwapper (no enhancer)", lambda r: r["generator"] == "inswapper" and r["has_faceenhancer"] == "False"),
        ("InSwapper (+faceenchancer)", lambda r: r["generator"] == "inswapper" and r["has_faceenhancer"] == "True"),
        ("UniFace (no enhancer)", lambda r: r["generator"] == "uniface" and r["has_faceenhancer"] == "False"),
        ("UniFace (+faceenchancer)", lambda r: r["generator"] == "uniface" and r["has_faceenhancer"] == "True"),
    ]
    picked = []
    for title, pred in wanted:
        match = next((r for r in test_rows if pred(r)), None)
        if match:
            item = dict(match)
            item["scenario"] = title
            picked.append(item)
    return picked


def main() -> None:
    print("=== 1. AUDIT TRẠNG THÁI CÁC MODULE TRONG LUỒNG DEMO ===")
    audit = audit_repo_stages()
    for stage, info in audit.items():
        mark = "READY" if info["ready"] else "BLOCKED"
        print(f"  [{mark:7s}] {stage:20s} -> {info['detail']}")

    print("\n=== 2. TẢI 3 CHECKPOINT M5 TỪ models/ & ĐO TỐC ĐỘ CPU APPLE M2 ===")
    det_siglip = DeepfakeDetector("king1oo1/deepfake-model", device="cpu", local_files_only=True)
    det_vit = DeepfakeDetector("prithivMLmods/Deep-Fake-Detector-v2-Model", device="cpu", local_files_only=True)
    det_cnn = LightweightCNNDetector(device="cpu")
    det_ens = EnsembleDeepfakeDetector(detectors=[det_siglip, det_vit], weights=[0.5, 0.5])

    print(f"  SigLIP2 adapter loaded: {det_siglip.adapter_loaded} | path={det_siglip.adapter_path} | tau={det_siglip.recommended_threshold}")
    print(f"  ViT     adapter loaded: {det_vit.adapter_loaded} | path={det_vit.adapter_path} | tau={det_vit.recommended_threshold}")
    print(f"  CNN     ckpt    loaded: True | path={det_cnn.checkpoint_path} | tau={det_cnn.recommended_threshold}")

    # Measure CPU batch=1 & batch=10 speed for all 4 M5 runtime options
    dummy_face = np.full((224, 224, 3), 127, dtype=np.uint8)
    dummy_crop = FaceCropResult(frame_id=0, has_face=True, bbox=(0, 0, 224, 224), det_confidence=0.99, face_rgb=dummy_face)
    batch10_crops = [
        FaceCropResult(frame_id=i, has_face=True, bbox=(0, 0, 224, 224), det_confidence=0.99, face_rgb=dummy_face)
        for i in range(10)
    ]

    cpu_benchmarks: Dict[str, Any] = {}
    for name, det in [
        ("finetuned_siglip2", det_siglip),
        ("finetuned_vit", det_vit),
        ("finetuned_efficientnet_b0", det_cnn),
        ("finetuned_ensemble_50_50", det_ens),
    ]:
        for _ in range(2):
            det.predict(dummy_crop)
        t1 = []
        for _ in range(12):
            ts = time.perf_counter()
            det.predict(dummy_crop)
            t1.append((time.perf_counter() - ts) * 1000.0)
        t10 = []
        for _ in range(4):
            ts = time.perf_counter()
            det.predict_batch(batch10_crops)
            t10.append((time.perf_counter() - ts) * 1000.0)
        m1, m10 = float(np.mean(t1)), float(np.mean(t10))
        cpu_benchmarks[name] = {
            "batch1_ms": round(m1, 2),
            "batch1_fps": round(1000.0 / m1, 2),
            "batch10_total_ms": round(m10, 2),
            "batch10_ms_per_frame": round(m10 / 10.0, 2),
            "batch10_fps": round(10000.0 / m10, 2),
        }
        print(
            f"  [CPU] {name:26s} -> batch1: {m1:6.1f} ms ({1000/m1:5.2f} FPS) | "
            f"batch10: {m10/10:5.1f} ms/fr ({10000/m10:5.2f} FPS)"
        )

    print("\n=== 3. CHẠY THỬ LUỒNG DEMO THẬT TRÊN 5 VIDEO `test` CHƯA THẤY (N=10, 5 FPS) ===")
    from facenet_pytorch import MTCNN

    mtcnn = MTCNN(keep_all=True, min_face_size=40, device="cpu")
    demo_videos = select_demo_test_videos(REPO_ROOT / "data" / "rtfs_m5_rich_metadata.csv")
    preview_dir = REPO_ROOT / "data" / "cache" / "m5_demo_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    e2e_runs = []
    for row in demo_videos:
        vpath = REPO_ROOT / row["filepath"]
        val_res = validate_video_input(
            filename=vpath.name,
            size_bytes=vpath.stat().st_size,
            video_path=str(vpath),
        )
        assert val_res.is_valid, f"Validator rejected {vpath.name}: {val_res.error_message}"

        t_total0 = time.perf_counter()

        # Stage 1: Frame extraction
        t_io0 = time.perf_counter()
        packets = extract_frames_demo(vpath, sample_fps=5, max_frames=10)
        ms_io = (time.perf_counter() - t_io0) * 1000.0

        # Stage 2: Face detection & crop
        t_fd0 = time.perf_counter()
        crops = [detect_largest_face_demo(mtcnn, p) for p in packets]
        ms_fd = (time.perf_counter() - t_fd0) * 1000.0

        # Stage 3: Anti-spoof pass-through stub (M4 not merged yet)
        valid_crops = [c for c in crops if c.has_face and c.face_rgb is not None]
        spoof_scores = [
            SpoofScore(frame_id=c.frame_id, probs={LABEL_REAL: 0.96, LABEL_PRINTED: 0.02, LABEL_ONSCREEN: 0.02})
            for c in valid_crops
        ]

        # Stage 4: REAL M5 Deepfake inference (primary SigLIP2 adapter)
        t_df0 = time.perf_counter()
        df_scores = det_siglip.predict_batch(valid_crops)
        ms_df = (time.perf_counter() - t_df0) * 1000.0

        # Stage 5: Decision DD6 + DD7
        final_label, confidence, agg_probs, reason = aggregate_dd6_demo(
            spoof_scores, df_scores, tau_spoof=0.60, tau_deepfake=det_siglip.recommended_threshold
        )

        # Stage 6: M6 UI visualization & stats (REAL app/components/visualizer.py)
        t_ui0 = time.perf_counter()
        frame_logs = []
        for c, sp, df in zip(valid_crops, spoof_scores, df_scores):
            frame_logs.append(
                {
                    "frame_id": c.frame_id,
                    "timestamp_sec": packets[c.frame_id].timestamp_sec,
                    "has_face": True,
                    "face_count": c.face_count,
                    "bbox": c.bbox,
                    "frame_label": LABEL_DEEPFAKE if df.fake_prob >= det_siglip.recommended_threshold else LABEL_REAL,
                    "spoof_probs": sp.probs,
                    "fake_prob": df.fake_prob,
                    "latency_ms": round((ms_fd + ms_df) / max(1, len(valid_crops)), 2),
                }
            )
        timeline = build_frame_timeline_rows(frame_logs)
        warnings = collect_decision_warnings(
            final_label=final_label,
            aggregated_probs=agg_probs,
            frame_logs=frame_logs,
            tau_spoof=0.60,
            tau_deepfake=det_siglip.recommended_threshold,
            uncertain_margin=0.05,
        )
        preview_crop = valid_crops[0]
        annotated = draw_face_annotation(
            frame_rgb=packets[preview_crop.frame_id].image_rgb,
            bbox=preview_crop.bbox,
            label=final_label,
            confidence=confidence,
        )
        preview_out = preview_dir / f"{row['sample_id']}_annotated.jpg"
        Image.fromarray(annotated, mode="RGB").save(preview_out, format="JPEG", quality=90)
        ms_ui = (time.perf_counter() - t_ui0) * 1000.0

        total_sec = time.perf_counter() - t_total0
        perf = compute_performance_stats(frame_logs, total_sec)

        # Extra comparison scores (outside primary pipeline latency clock)
        df_scores_cnn = det_cnn.predict_batch(valid_crops)
        df_scores_ens = det_ens.predict_batch(valid_crops)

        cnn_prob = float(np.mean([d.fake_prob for d in df_scores_cnn]))
        ens_prob = float(np.mean([d.fake_prob for d in df_scores_ens]))
        correct = final_label == row["label"]
        e2e_runs.append(
            {
                "scenario": row["scenario"],
                "sample_id": row["sample_id"],
                "true_label": row["label"],
                "pred_label_siglip2": final_label,
                "correct": correct,
                "siglip2_fake_prob": agg_probs[LABEL_DEEPFAKE],
                "cnn_fake_prob": round(cnn_prob, 4),
                "ensemble_fake_prob": round(ens_prob, 4),
                "n_frames": len(valid_crops),
                "ms_video_io": round(ms_io, 1),
                "ms_face_det": round(ms_fd, 1),
                "ms_m5_deepfake": round(ms_df, 1),
                "ms_ui_render": round(ms_ui, 1),
                "total_sec": round(total_sec, 2),
                "effective_e2e_fps": perf["effective_fps"],
                "warnings": warnings,
                "timeline_len": len(timeline),
            }
        )
        status_str = "PASS" if correct else "MISS"
        print(
            f"  [{status_str}] {row['scenario']:27s} | True={row['label']:8s} -> Pred={final_label:8s} "
            f"(SigLIP2={agg_probs[LABEL_DEEPFAKE]:.3f}, CNN={cnn_prob:.3f}, Ens={ens_prob:.3f}) | "
            f"io={ms_io:.0f}ms + fd={ms_fd:.0f}ms + m5={ms_df:.0f}ms + ui={ms_ui:.0f}ms "
            f"= {total_sec:.2f}s ({perf['effective_fps']:.2f} FPS)"
        )

    out_path = REPO_ROOT / "results" / "m5_e2e_demo_verification.json"
    out_path.write_text(
        json.dumps(
            {"module_audit": audit, "cpu_benchmarks": cpu_benchmarks, "e2e_test_videos": e2e_runs},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved E2E verification report -> {out_path}")


if __name__ == "__main__":
    main()
