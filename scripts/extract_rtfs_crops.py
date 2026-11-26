"""Fast multi-process MTCNN face crop extraction for all 800 RTFS-800 videos.

Extracts `frames_per_video` evenly spaced RGB frames per video, detects the largest face
with MTCNN + 25% context margin, resizes to (224, 224, 3) RGB uint8, and caches PNGs
under `data/cache/rtfs_all_f<N>/<sample_id>/`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]

_WORKER_MTCNN = None


def _init_worker() -> None:
    global _WORKER_MTCNN
    import torch
    from facenet_pytorch import MTCNN

    # Avoid CPU thread oversubscription across worker processes
    torch.set_num_threads(2)
    _WORKER_MTCNN = MTCNN(keep_all=True, min_face_size=40, device="cpu")


def _detect_largest_face(
    frame_rgb: np.ndarray,
    margin_ratio: float = 0.25,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray | None:
    assert _WORKER_MTCNN is not None
    h, w = frame_rgb.shape[:2]
    scale = 1.0
    max_dim = max(h, w)
    if max_dim > 640:
        scale = 640.0 / max_dim
        small = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = frame_rgb

    boxes, probs = _WORKER_MTCNN.detect(Image.fromarray(small, mode="RGB"))
    if boxes is None or len(boxes) == 0:
        return None

    valid = [b for b, p in zip(boxes, probs) if p is not None and float(p) >= 0.80] or list(boxes)
    best = max(valid, key=lambda b: max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1])))
    x1 = int(round(float(best[0]) / scale))
    y1 = int(round(float(best[1]) / scale))
    x2 = int(round(float(best[2]) / scale))
    y2 = int(round(float(best[3]) / scale))
    if x2 <= x1 or y2 <= y1:
        return None

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin_ratio)
    nx1 = max(0, int(round(cx - side / 2.0)))
    ny1 = max(0, int(round(cy - side / 2.0)))
    nx2 = min(w, int(round(cx + side / 2.0)))
    ny2 = min(h, int(round(cy + side / 2.0)))

    crop = frame_rgb[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None
    return np.ascontiguousarray(cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR), dtype=np.uint8)


def _process_one_video(args: Tuple[str, str, str, int]) -> Tuple[str, int, bool, bool]:
    sample_id, video_path_str, out_dir_str, frames_per_video = args
    sample_dir = Path(out_dir_str)
    done_flag = sample_dir / "done.json"
    if done_flag.is_file():
        meta = json.loads(done_flag.read_text(encoding="utf-8"))
        return sample_id, int(meta.get("n_crops", 0)), bool(meta.get("used_fallback", False)), True

    sample_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path_str)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

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

    crops: List[np.ndarray] = []
    raw_fallback: List[np.ndarray] = []

    for f_idx in indices:
        if len(crops) >= frames_per_video:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if len(raw_fallback) < frames_per_video:
            raw_fallback.append(rgb)
        face = _detect_largest_face(rgb)
        if face is not None:
            crops.append(face)

    cap.release()

    used_fallback = False
    if not crops and raw_fallback:
        used_fallback = True
        for rgb in raw_fallback[:frames_per_video]:
            fh, fw = rgb.shape[:2]
            side = int(min(fh, fw) * 0.65)
            y1 = max(0, (fh - side) // 2)
            x1 = max(0, (fw - side) // 2)
            sq = rgb[y1 : y1 + side, x1 : x1 + side]
            crops.append(cv2.resize(sq, (224, 224), interpolation=cv2.INTER_LINEAR))

    for k, arr in enumerate(crops):
        Image.fromarray(arr, mode="RGB").save(sample_dir / f"frame_{k:02d}.png")
    done_flag.write_text(
        json.dumps({"n_crops": len(crops), "used_fallback": used_fallback}),
        encoding="utf-8",
    )
    return sample_id, len(crops), used_fallback, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel MTCNN face crop extraction for RTFS-800.")
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data" / "rtfs_m5_rich_metadata.csv")
    parser.add_argument("--frames-per-video", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    cache_dir = args.cache_dir or (REPO_ROOT / "data" / "cache" / f"rtfs_all_f{args.frames_per_video}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    with args.metadata.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    tasks = [
        (r["sample_id"], str(REPO_ROOT / r["filepath"]), str(cache_dir / r["sample_id"]), args.frames_per_video)
        for r in rows
    ]

    t0 = time.perf_counter()
    total_crops = 0
    fallbacks = 0
    cached_hits = 0

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = [pool.submit(_process_one_video, task) for task in tasks]
        for done_idx, fut in enumerate(as_completed(futures), start=1):
            _sid, n_crops, used_fb, was_cached = fut.result()
            total_crops += n_crops
            fallbacks += int(used_fb)
            cached_hits += int(was_cached)
            if done_idx % 80 == 0 or done_idx == len(tasks):
                print(
                    f"  [MTCNN x{args.workers}] {done_idx}/{len(tasks)} videos "
                    f"({total_crops} crops, fallbacks={fallbacks}, elapsed={time.perf_counter() - t0:.1f}s)"
                )

    print(
        f"Done: {total_crops} crops across {len(tasks)} videos in {time.perf_counter() - t0:.1f}s "
        f"(cache_hits={cached_hits}, fallbacks={fallbacks}) -> {cache_dir}"
    )


if __name__ == "__main__":
    main()
