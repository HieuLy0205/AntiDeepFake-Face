"""Benchmark pretrained Stage 2 Deepfake models on dataset_RTFS_800_Subset (M5).

Workflow:
1. Reads `data/rtfs_m5_rich_metadata.csv` (strictly leak-free val/test split by `subject_id`).
2. Extracts evenly-spaced RGB frames per video, detects the largest face (matching M3 rules),
   crops with context margin, resizes to (224, 224, 3) RGB uint8, and caches crops under
   `data/cache/rtfs_<split>_crops/` so every candidate model runs on identical inputs.
3. Evaluates each candidate model on CPU:
   - Video-level ROC-AUC (with 95% bootstrap CI), PR-AUC, EER, optimal threshold tau_d,
     Accuracy, Balanced Accuracy, F1-macro, APCER, BPCER, ACER, Confusion Matrix.
   - Slice-based analysis: InSwapper vs UniFace, without vs with `faceenchancer`,
     lighting (`bright`/`average`/`dark`), gender (`man`/`woman`).
   - Operational CPU latency (batch=1 single-frame latency & effective FPS; batch=N latency).
4. Saves reproducible JSON and per-video prediction CSV under `results/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deepfake.predictor import DeepfakeDetector  # noqa: E402
from src.schemas import FaceCropResult  # noqa: E402


DEFAULT_CANDIDATES = [
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "prithivMLmods/Deepfake-Detect-Siglip2",
    "dima806/deepfake_vs_real_image_detection",
    "king1oo1/deepfake-model",
]


class FaceCropper:
    """CPU face detector (MTCNN primary, OpenCV Haar fallback) returning largest 224x224 RGB crop."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224), margin_ratio: float = 0.25) -> None:
        self.target_size = target_size
        self.margin_ratio = margin_ratio
        self.backend = "none"
        self.mtcnn = None
        self.cascades = []

        try:
            from facenet_pytorch import MTCNN  # type: ignore

            self.mtcnn = MTCNN(keep_all=True, min_face_size=40, device="cpu")
            self.backend = "mtcnn"
            return
        except Exception:
            pass

        cascade_dir = Path(cv2.data.haarcascades)
        for fname in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"):
            p = cascade_dir / fname
            if p.is_file():
                clf = cv2.CascadeClassifier(str(p))
                if not clf.empty():
                    self.cascades.append(clf)
        if self.cascades:
            self.backend = "haar"
        else:
            raise RuntimeError("Neither facenet-pytorch MTCNN nor OpenCV Haar cascades are available.")

    def detect_and_crop(
        self, frame_rgb: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        h, w = frame_rgb.shape[:2]
        scale = 1.0
        max_dim = max(h, w)
        if max_dim > 640:
            scale = 640.0 / max_dim
            small = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = frame_rgb

        x1 = y1 = x2 = y2 = 0
        found = False

        if self.backend == "mtcnn" and self.mtcnn is not None:
            pil_small = Image.fromarray(small, mode="RGB")
            boxes, probs = self.mtcnn.detect(pil_small)
            if boxes is not None and len(boxes) > 0:
                valid = [
                    b for b, p in zip(boxes, probs) if p is not None and float(p) >= 0.80
                ] or list(boxes)
                best = max(valid, key=lambda b: max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1])))
                x1 = int(round(float(best[0]) / scale))
                y1 = int(round(float(best[1]) / scale))
                x2 = int(round(float(best[2]) / scale))
                y2 = int(round(float(best[3]) / scale))
                found = (x2 > x1) and (y2 > y1)
        else:
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
            boxes = []
            for clf in self.cascades:
                detected = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
                if len(detected) > 0:
                    boxes = detected
                    break
            if len(boxes) > 0:
                bx, by, bw, bh = max(boxes, key=lambda b: int(b[2]) * int(b[3]))
                x1 = int(round(bx / scale))
                y1 = int(round(by / scale))
                x2 = int(round((bx + bw) / scale))
                y2 = int(round((by + bh) / scale))
                found = True

        if not found:
            return False, None, None

        # Expand square crop with context margin so jaw/blending boundaries are preserved
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * self.margin_ratio)
        nx1 = max(0, int(round(cx - side / 2.0)))
        ny1 = max(0, int(round(cy - side / 2.0)))
        nx2 = min(w, int(round(cx + side / 2.0)))
        ny2 = min(h, int(round(cy + side / 2.0)))

        crop = frame_rgb[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            return False, None, None
        resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        return True, np.ascontiguousarray(resized, dtype=np.uint8), (nx1, ny1, nx2, ny2)


def extract_or_load_crops(
    rows: List[dict],
    repo_root: Path,
    cache_dir: Path,
    frames_per_video: int = 5,
) -> Dict[str, List[np.ndarray]]:
    """Extract and cache 224x224 RGB uint8 face crops for each video."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cropper = FaceCropper(target_size=(224, 224), margin_ratio=0.25)
    print(f"  [Crop] Face detector backend: {cropper.backend}")

    crops_by_sample: Dict[str, List[np.ndarray]] = {}
    fallback_center_count = 0
    t0 = time.perf_counter()

    for idx, row in enumerate(rows, start=1):
        sid = row["sample_id"]
        sample_dir = cache_dir / sid
        meta_flag = sample_dir / "done.json"

        if meta_flag.is_file():
            imgs = []
            for img_p in sorted(sample_dir.glob("*.png")):
                with Image.open(img_p) as im:
                    imgs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
            crops_by_sample[sid] = imgs
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)
        video_path = repo_root / row["filepath"]
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Sample candidate positions evenly (oversample 2x so if a frame misses face we still get N)
        n_probe = max(frames_per_video * 2, frames_per_video + 4)
        if total_frames > 0:
            indices = np.linspace(
                int(total_frames * 0.08),
                max(0, int(total_frames * 0.92) - 1),
                num=min(n_probe, total_frames),
                dtype=int,
            ).tolist()
        else:
            indices = list(range(n_probe))

        detected_crops: List[np.ndarray] = []
        raw_fallback_frames: List[np.ndarray] = []

        for f_idx in indices:
            if len(detected_crops) >= frames_per_video:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if len(raw_fallback_frames) < frames_per_video:
                raw_fallback_frames.append(rgb)
            has_face, face_rgb, _bbox = cropper.detect_and_crop(rgb)
            if has_face and face_rgb is not None:
                detected_crops.append(face_rgb)

        cap.release()

        used_fallback = False
        if not detected_crops and raw_fallback_frames:
            # Center portrait square fallback only if Haar missed rotated/dark face
            used_fallback = True
            fallback_center_count += 1
            for rgb in raw_fallback_frames[:frames_per_video]:
                fh, fw = rgb.shape[:2]
                side = int(min(fh, fw) * 0.65)
                y1 = max(0, (fh - side) // 2)
                x1 = max(0, (fw - side) // 2)
                sq = rgb[y1 : y1 + side, x1 : x1 + side]
                detected_crops.append(cv2.resize(sq, (224, 224), interpolation=cv2.INTER_LINEAR))

        for k, arr in enumerate(detected_crops):
            Image.fromarray(arr, mode="RGB").save(sample_dir / f"frame_{k:02d}.png")
        meta_flag.write_text(
            json.dumps({"n_crops": len(detected_crops), "used_fallback": used_fallback}),
            encoding="utf-8",
        )
        crops_by_sample[sid] = detected_crops

        if idx % 40 == 0 or idx == len(rows):
            print(f"  [Crop] {idx}/{len(rows)} videos processed ({time.perf_counter() - t0:.1f}s)")

    if fallback_center_count > 0:
        print(f"  [Crop] Note: {fallback_center_count}/{len(rows)} videos used center-crop fallback.")
    return crops_by_sample


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(float(roc_auc_score(yt, y_score[idx])))
    if not aucs:
        return (0.0, 0.0)
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Find threshold minimizing ACER = (APCER + BPCER)/2 (equivalent to maximizing Youden J)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # BPCER = fpr (Real misclassified as Fake), APCER = 1 - tpr (Fake misclassified as Real)
    acers = 0.5 * (fpr + (1.0 - tpr))
    best_idx = int(np.argmin(acers))
    eer_idx = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    eer = float(0.5 * (fpr[eer_idx] + (1.0 - tpr[eer_idx])))
    best_thr = float(thresholds[best_idx])
    if not np.isfinite(best_thr):
        best_thr = 0.5
    return best_thr, eer


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # In biometric ISO/IEC 30107-3:
    # Real=0 (Bona fide), Deepfake=1 (Attack)
    # APCER = FN / (TP + FN) [attacks falsely accepted as Real]
    # BPCER = FP / (TN + FP) [real falsely rejected as Attack]
    apcer = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    bpcer = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    acer = 0.5 * (apcer + bpcer)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_fake": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "apcer": round(apcer, 4),
        "bpcer": round(bpcer, 4),
        "acer": round(acer, 4),
        "confusion_matrix": {"tn_real": int(tn), "fp_real_as_fake": int(fp), "fn_fake_as_real": int(fn), "tp_fake": int(tp)},
    }


def benchmark_model(
    model_id: str,
    rows: List[dict],
    crops_by_sample: Dict[str, List[np.ndarray]],
    batch_size: int = 16,
    fixed_threshold: Optional[float] = None,
    local_files_only: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    print(f"\n=== Benchmarking: {model_id} ===")
    t_load0 = time.perf_counter()
    detector = DeepfakeDetector(
        model_id=model_id,
        device="cpu",
        backend="pytorch",
        use_default_adapter=False,
        local_files_only=local_files_only,
    )
    load_sec = time.perf_counter() - t_load0

    num_params_m = sum(p.numel() for p in detector._model.parameters()) / 1e6
    print(f"  Loaded in {load_sec:.2f}s | params={num_params_m:.1f}M | id2label={detector.id2label} | fake_idx={detector.fake_label_index}")

    # Measure realistic single-frame CPU latency (batch=1, 20 runs after 3 warmup runs)
    sample_face = next(iter(crops_by_sample.values()))[0]
    dummy_crop = FaceCropResult(frame_id=0, has_face=True, bbox=(0, 0, 224, 224), det_confidence=0.99, face_rgb=sample_face)
    for _ in range(3):
        detector.predict(dummy_crop)
    single_times_ms = []
    for _ in range(20):
        ts = time.perf_counter()
        detector.predict(dummy_crop)
        single_times_ms.append((time.perf_counter() - ts) * 1000.0)

    # Measure batch=10 latency (matches N=10 video mode in UI/E6)
    batch10_crops = [
        FaceCropResult(frame_id=i, has_face=True, bbox=(0, 0, 224, 224), det_confidence=0.99, face_rgb=sample_face)
        for i in range(10)
    ]
    batch10_times_ms = []
    for _ in range(5):
        ts = time.perf_counter()
        detector.predict_batch(batch10_crops)
        batch10_times_ms.append((time.perf_counter() - ts) * 1000.0)

    # Run full evaluation across all videos
    t_eval0 = time.perf_counter()
    flat_items: List[Tuple[str, np.ndarray]] = []
    for row in rows:
        sid = row["sample_id"]
        for arr in crops_by_sample[sid]:
            flat_items.append((sid, arr))

    frame_scores_by_sample: Dict[str, List[float]] = {r["sample_id"]: [] for r in rows}
    for i in range(0, len(flat_items), batch_size):
        chunk = flat_items[i : i + batch_size]
        probs = detector.predict_probs_raw([x[1] for x in chunk])
        for (sid, _), p in zip(chunk, probs):
            frame_scores_by_sample[sid].append(p)
        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, len(flat_items))
            print(f"  [Eval] {done}/{len(flat_items)} frames scored ({time.perf_counter() - t_eval0:.1f}s)")

    eval_sec = time.perf_counter() - t_eval0

    # Aggregate frame scores -> video score (mean per DD7)
    pred_rows: List[Dict[str, Any]] = []
    y_true_list, y_score_list = [], []
    for row in rows:
        sid = row["sample_id"]
        f_scores = frame_scores_by_sample[sid]
        mean_score = float(np.mean(f_scores)) if f_scores else 0.0
        std_score = float(np.std(f_scores)) if f_scores else 0.0
        y = 1 if row["label"] == "Deepfake" else 0
        y_true_list.append(y)
        y_score_list.append(mean_score)
        pred_rows.append(
            {
                "model_id": model_id,
                "sample_id": sid,
                "split": row["split"],
                "label": row["label"],
                "y_true": y,
                "fake_prob_mean": round(mean_score, 6),
                "fake_prob_std": round(std_score, 6),
                "n_frames": len(f_scores),
                "generator": row["generator"],
                "has_faceenhancer": row["has_faceenhancer"],
                "lighting": row["lighting"],
                "gender": row["gender"],
            }
        )

    y_true = np.asarray(y_true_list, dtype=int)
    y_score = np.asarray(y_score_list, dtype=float)

    auc = float(roc_auc_score(y_true, y_score))
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_score, n_boot=1000, seed=42)
    pr_auc = float(average_precision_score(y_true, y_score))
    opt_thr, eer = find_optimal_threshold(y_true, y_score)
    eval_thr = fixed_threshold if fixed_threshold is not None else opt_thr

    metrics_default = compute_metrics_at_threshold(y_true, y_score, threshold=0.50)
    metrics_at_tau = compute_metrics_at_threshold(y_true, y_score, threshold=eval_thr)

    # Slice-based analysis (ml-best-practices)
    def slice_auc(mask_fn) -> Dict[str, Any]:
        sub = [p for p in pred_rows if mask_fn(p)]
        yt = np.asarray([p["y_true"] for p in sub], dtype=int)
        ys = np.asarray([p["fake_prob_mean"] for p in sub], dtype=float)
        out: Dict[str, Any] = {
            "n_samples": len(sub),
            "n_real": int(np.sum(yt == 0)),
            "n_fake": int(np.sum(yt == 1)),
            "mean_fake_prob_real": round(float(np.mean(ys[yt == 0])), 4) if np.any(yt == 0) else None,
            "mean_fake_prob_fake": round(float(np.mean(ys[yt == 1])), 4) if np.any(yt == 1) else None,
        }
        if len(np.unique(yt)) == 2:
            out["roc_auc"] = round(float(roc_auc_score(yt, ys)), 4)
            m = compute_metrics_at_threshold(yt, ys, threshold=eval_thr)
            out["apcer"] = m["apcer"]
            out["bpcer"] = m["bpcer"]
            out["acer"] = m["acer"]
            out["recall_fake"] = round(1.0 - m["apcer"], 4)
        return out

    slices = {
        "real_vs_inswapper_all": slice_auc(lambda p: p["generator"] in ("original", "inswapper")),
        "real_vs_inswapper_no_enhancer": slice_auc(
            lambda p: p["generator"] == "original" or (p["generator"] == "inswapper" and p["has_faceenhancer"] == "False")
        ),
        "real_vs_inswapper_with_enhancer": slice_auc(
            lambda p: p["generator"] == "original" or (p["generator"] == "inswapper" and p["has_faceenhancer"] == "True")
        ),
        "real_vs_uniface_all": slice_auc(lambda p: p["generator"] in ("original", "uniface")),
        "real_vs_uniface_no_enhancer": slice_auc(
            lambda p: p["generator"] == "original" or (p["generator"] == "uniface" and p["has_faceenhancer"] == "False")
        ),
        "real_vs_uniface_with_enhancer": slice_auc(
            lambda p: p["generator"] == "original" or (p["generator"] == "uniface" and p["has_faceenhancer"] == "True")
        ),
        "all_fake_no_enhancer": slice_auc(
            lambda p: p["generator"] == "original" or p["has_faceenhancer"] == "False"
        ),
        "all_fake_with_enhancer": slice_auc(
            lambda p: p["generator"] == "original" or p["has_faceenhancer"] == "True"
        ),
        "lighting_bright": slice_auc(lambda p: p["lighting"] == "bright"),
        "lighting_average": slice_auc(lambda p: p["lighting"] == "average"),
        "lighting_dark": slice_auc(lambda p: p["lighting"] == "dark"),
        "gender_man": slice_auc(lambda p: p["gender"] == "man"),
        "gender_woman": slice_auc(lambda p: p["gender"] == "woman"),
    }

    mean_single_ms = float(np.mean(single_times_ms))
    p95_single_ms = float(np.percentile(single_times_ms, 95))
    mean_b10_ms = float(np.mean(batch10_times_ms))

    summary = {
        "model_id": model_id,
        "id2label": detector.id2label,
        "fake_label_index": detector.fake_label_index,
        "params_millions": round(num_params_m, 2),
        "load_time_sec": round(load_sec, 2),
        "total_eval_sec": round(eval_sec, 2),
        "cpu_latency": {
            "batch1_mean_ms": round(mean_single_ms, 2),
            "batch1_p95_ms": round(p95_single_ms, 2),
            "batch1_fps": round(1000.0 / mean_single_ms, 2),
            "batch10_total_ms": round(mean_b10_ms, 2),
            "batch10_ms_per_frame": round(mean_b10_ms / 10.0, 2),
            "batch10_fps": round(10000.0 / mean_b10_ms, 2),
        },
        "overall": {
            "n_videos": len(rows),
            "n_real": int(np.sum(y_true == 0)),
            "n_fake": int(np.sum(y_true == 1)),
            "roc_auc": round(auc, 4),
            "roc_auc_ci95": [round(ci_low, 4), round(ci_high, 4)],
            "pr_auc": round(pr_auc, 4),
            "eer": round(eer, 4),
            "optimal_threshold_val": round(opt_thr, 4),
            "at_0_50": metrics_default,
            "at_selected_tau": metrics_at_tau,
        },
        "slices": slices,
    }
    print(
        f"  => ROC-AUC={summary['overall']['roc_auc']} (95% CI {summary['overall']['roc_auc_ci95']}) | "
        f"EER={summary['overall']['eer']} | ACER@opt({eval_thr:.3f})={metrics_at_tau['acer']} | "
        f"CPU batch1={mean_single_ms:.1f}ms ({1000.0/mean_single_ms:.2f} FPS), "
        f"batch10={mean_b10_ms/10.0:.1f}ms/fr ({10000.0/mean_b10_ms:.2f} FPS)"
    )
    return summary, pred_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pretrained M5 deepfake models on RTFS-800.")
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data" / "rtfs_m5_rich_metadata.csv")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--models", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--frames-per-video", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-videos", type=int, default=0, help="Optional cap for quick smoke tests (0=all).")
    parser.add_argument("--fixed-threshold", type=float, default=None, help="Apply pre-selected val threshold on test.")
    parser.add_argument("--extract-only", action="store_true", help="Only extract and cache 224x224 face crops.")
    parser.add_argument("--local-only", action="store_true", help="Load models from local cache only.")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-preds", type=Path, default=None)
    args = parser.parse_args()

    with args.metadata.open("r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == args.split]

    if args.max_videos > 0:
        rows = rows[: args.max_videos]

    print(f"Loaded {len(rows)} videos for split='{args.split}' from {args.metadata}")
    cache_dir = REPO_ROOT / "data" / "cache" / f"rtfs_{args.split}_f{args.frames_per_video}"
    crops_by_sample = extract_or_load_crops(
        rows=rows,
        repo_root=REPO_ROOT,
        cache_dir=cache_dir,
        frames_per_video=args.frames_per_video,
    )
    if args.extract_only:
        total_crops = sum(len(v) for v in crops_by_sample.values())
        print(f"Extract-only complete: {total_crops} face crops cached at {cache_dir}")
        return

    all_summaries: List[Dict[str, Any]] = []
    all_preds: List[Dict[str, Any]] = []

    for model_id in args.models:
        try:
            summary, preds = benchmark_model(
                model_id=model_id,
                rows=rows,
                crops_by_sample=crops_by_sample,
                batch_size=args.batch_size,
                fixed_threshold=args.fixed_threshold,
                local_files_only=args.local_only,
            )
            all_summaries.append(summary)
            all_preds.extend(preds)
        except Exception as exc:
            print(f"  [ERROR] Model {model_id} failed: {type(exc).__name__}: {exc}")
            all_summaries.append({"model_id": model_id, "error": f"{type(exc).__name__}: {exc}"})

    out_json = args.out_json or (REPO_ROOT / "results" / f"m5_{args.split}_benchmark.json")
    out_preds = args.out_preds or (REPO_ROOT / "results" / f"m5_{args.split}_predictions.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "split": args.split,
                "frames_per_video": args.frames_per_video,
                "n_videos": len(rows),
                "models": all_summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if all_preds:
        with out_preds.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_preds[0].keys()))
            writer.writeheader()
            writer.writerows(all_preds)

    print(f"\nSaved benchmark JSON -> {out_json}")
    print(f"Saved predictions CSV -> {out_preds}")


if __name__ == "__main__":
    main()
