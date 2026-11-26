"""Prepare leak-free val/test metadata from dataset_RTFS_800_Subset for M5 & M2.

Generates:
1. data/metadata.csv — strictly the 7 frozen columns required by the project design:
   (sample_id, filepath, label, subject_id, split, source, device)
2. data/rtfs_m5_rich_metadata.csv — extended metadata with generator, faceenhancer flag,
   lighting/location/gender tags, source photo, and video technical specs for slice-based
   error analysis.

Leakage prevention:
- Groups all videos sharing the same source YouTube video ID (`subject_id`) so no
  original video / speaker appears in both `val` and `test`.
- Stratified by class (`Real`, `inswapper`, `uniface`) to preserve class ratios across splits.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import random


FROZEN_COLUMNS = [
    "sample_id",
    "filepath",
    "label",
    "subject_id",
    "split",
    "source",
    "device",
]

RICH_COLUMNS = FROZEN_COLUMNS + [
    "generator",
    "has_faceenhancer",
    "target_clip",
    "source_photo",
    "lighting",
    "location",
    "gender",
    "tags",
    "duration_sec",
    "width",
    "height",
    "fps",
    "size_mb",
]


def _parse_tags(tag_str: str) -> Tuple[str, str, str]:
    tokens = {t.strip().lower() for t in (tag_str or "").split(";") if t.strip()}
    # Fix known typo 'intdoor' in RTFS-10k CSVs
    if "intdoor" in tokens:
        tokens.add("indoor")
    lighting = next((x for x in ("dark", "average", "bright") if x in tokens), "unknown")
    location = next((x for x in ("indoor", "outdoor") if x in tokens), "unknown")
    gender = next((x for x in ("woman", "man", "child") if x in tokens), "unknown")
    return lighting, location, gender


def _probe_video(path: Path) -> Tuple[float, int, int, float]:
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return 0.0, 0, 0, 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fc = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
        dur = round(fc / fps, 2) if fps > 0 and fc > 0 else 0.0
        return dur, w, h, round(fps, 2)
    except Exception:
        return 0.0, 0, 0, 0.0


def _load_csv_index(csv_path: Path, key_col: str) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            key = (row.get(key_col) or "").strip()
            if key:
                index[key] = row
    return index


def _assign_group_splits(
    records: List[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, str]:
    """Assign each subject_id (YouTube video ID) to train/val/test with zero leakage.

    Supports both 2-way (train_ratio=0.0 -> val/test) and 3-way (train/val/test).
    Groups are shuffled deterministically and greedily assigned to match stratified
    target counts per generator while guaranteeing zero subject_id overlap across splits.
    """
    group_items: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        group_items[rec["subject_id"]].append(rec)

    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    ratios: Dict[str, float] = {}
    if train_ratio > 0:
        ratios["train"] = train_ratio
    if val_ratio > 0:
        ratios["val"] = val_ratio
    if test_ratio > 0:
        ratios["test"] = test_ratio

    total_by_bucket = Counter(rec["generator"] for rec in records)
    targets = {
        sp: {b: total_by_bucket[b] * r for b, r in zip(total_by_bucket.keys(), [r] * len(total_by_bucket))}
        for sp, r in ratios.items()
    }
    current = {sp: Counter() for sp in ratios}

    groups = sorted(group_items.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)

    group_split: Dict[str, str] = {}
    split_names = list(ratios.keys())

    for g in groups:
        g_counts = Counter(rec["generator"] for rec in group_items[g])
        best_split = split_names[0]
        best_score = float("inf")
        for candidate in split_names:
            # Normalized squared deficit across all splits if g is placed in `candidate`
            score = 0.0
            for sp in split_names:
                for b in total_by_bucket:
                    new_cnt = current[sp][b] + (g_counts[b] if sp == candidate else 0)
                    tgt = max(1.0, targets[sp][b])
                    diff = (new_cnt - targets[sp][b]) / tgt
                    score += diff * diff
            if score < best_score:
                best_score = score
                best_split = candidate
        group_split[g] = best_split
        current[best_split].update(g_counts)

    return group_split


def build_metadata(
    dataset_root: Path,
    repo_root: Path,
    train_ratio: float = 0.60,
    val_ratio: float = 0.15,
    seed: int = 42,
    probe_videos: bool = True,
) -> List[dict]:
    videos_idx = _load_csv_index(dataset_root / "videos.csv", "filename")
    inswapper_idx = _load_csv_index(dataset_root / "inswapper.csv", "filename")
    uniface_idx = _load_csv_index(dataset_root / "uniface.csv", "filename")

    sampled_root = dataset_root / "data_sampled"
    records: List[dict] = []

    specs = [
        ("original_videos", "Real", "original", videos_idx),
        ("inswapper", "Deepfake", "inswapper", inswapper_idx),
        ("uniface", "Deepfake", "uniface", uniface_idx),
    ]

    for subdir, label, generator, idx in specs:
        folder = sampled_root / subdir
        mp4_files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")
        for seq_num, mp4_path in enumerate(mp4_files, start=1):
            fn = mp4_path.name
            row = idx.get(fn, {})

            if generator == "original":
                target_clip = fn
                yt_id = (row.get("id") or fn[:11]).strip()
                source_photo = ""
                has_enh = False
                tag_str = (row.get("tags") or "").strip()
                source_name = "rtfs10k_original"
            else:
                target_clip = (row.get("video") or "").strip()
                # Map target_clip -> videos.csv['id'] so leading '-' YouTube IDs match exactly
                orig_row = videos_idx.get(target_clip, {})
                yt_id = (orig_row.get("id") or target_clip[:11] or fn[:11]).strip()
                source_photo = (row.get("photo") or "").strip()
                add_feat = (row.get("additional_features") or "").strip().lower()
                has_enh = "enhancer" in add_feat or "enchancer" in add_feat
                tag_str = (row.get("video_tags") or orig_row.get("tags") or "").strip()
                source_name = f"rtfs10k_{generator}_enhancer" if has_enh else f"rtfs10k_{generator}"

            lighting, location, gender = _parse_tags(tag_str)
            try:
                rel_path = mp4_path.resolve().relative_to(repo_root.resolve()).as_posix()
            except ValueError:
                rel_path = mp4_path.as_posix()

            dur, w, h, fps = _probe_video(mp4_path) if probe_videos else (0.0, 0, 0, 0.0)
            size_mb = round(mp4_path.stat().st_size / (1024 * 1024), 3)

            records.append(
                {
                    "sample_id": f"rtfs_{generator}_{seq_num:04d}",
                    "filepath": rel_path,
                    "label": label,
                    "subject_id": f"yt_{yt_id}",
                    "split": "",
                    "source": source_name,
                    "device": "youtube_videocall",
                    "generator": generator,
                    "has_faceenhancer": str(has_enh),
                    "target_clip": target_clip,
                    "source_photo": source_photo,
                    "lighting": lighting,
                    "location": location,
                    "gender": gender,
                    "tags": tag_str,
                    "duration_sec": dur,
                    "width": w,
                    "height": h,
                    "fps": fps,
                    "size_mb": size_mb,
                }
            )

    group_split = _assign_group_splits(
        records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    for rec in records:
        rec["split"] = group_split[rec["subject_id"]]

    # Verify zero subject_id leakage across all pairs of splits
    by_split: Dict[str, Set[str]] = defaultdict(set)
    for r in records:
        by_split[r["split"]].add(r["subject_id"])
    split_keys = sorted(by_split.keys())
    for i in range(len(split_keys)):
        for j in range(i + 1, len(split_keys)):
            a, b = split_keys[i], split_keys[j]
            overlap = by_split[a] & by_split[b]
            if overlap:
                raise RuntimeError(f"Subject leakage detected between {a} and {b}: {len(overlap)} IDs")

    return records


def write_csvs(records: List[dict], out_standard: Path, out_rich: Path) -> None:
    out_standard.parent.mkdir(parents=True, exist_ok=True)
    with out_standard.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FROZEN_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    out_rich.parent.mkdir(parents=True, exist_ok=True)
    with out_rich.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RICH_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build leak-free val/test metadata for RTFS-800.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "dataset_RTFS_800_Subset",
        help="Path to dataset_RTFS_800_Subset directory.",
    )
    parser.add_argument(
        "--out-metadata",
        type=Path,
        default=repo_root / "data" / "metadata.csv",
        help="Output 7-column project standard metadata CSV.",
    )
    parser.add_argument(
        "--out-rich",
        type=Path,
        default=repo_root / "data" / "rtfs_m5_rich_metadata.csv",
        help="Output rich M5 metadata CSV for slice-based error analysis.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.60, help="Fraction for train (0.0 for 2-way val/test).")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction for val.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-probe", action="store_true", help="Skip OpenCV duration/resolution probing.")
    args = parser.parse_args()

    records = build_metadata(
        dataset_root=args.dataset_root,
        repo_root=repo_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        probe_videos=not args.no_probe,
    )
    write_csvs(records, args.out_metadata, args.out_rich)

    # Print summary report
    print(f"Wrote {len(records)} rows to {args.out_metadata} and {args.out_rich}")
    active_splits = [sp for sp in ("train", "val", "test") if any(r["split"] == sp for r in records)]
    for split in active_splits:
        sub = [r for r in records if r["split"] == split]
        labels = Counter(r["label"] for r in sub)
        sources = Counter(r["source"] for r in sub)
        subjects = len({r["subject_id"] for r in sub})
        print(f"  [{split:5s}] total={len(sub):3d}, unique_subjects={subjects:3d}, labels={dict(labels)}")
        print(f"          sources={dict(sorted(sources.items()))}")

    for i in range(len(active_splits)):
        for j in range(i + 1, len(active_splits)):
            a, b = active_splits[i], active_splits[j]
            sa = {r["subject_id"] for r in records if r["split"] == a}
            sb = {r["subject_id"] for r in records if r["split"] == b}
            ca = {r["target_clip"] for r in records if r["split"] == a}
            cb = {r["target_clip"] for r in records if r["split"] == b}
            print(f"  Leakage check ({a} vs {b}) -> subject_id overlap: {len(sa & sb)}, target_clip overlap: {len(ca & cb)}")


if __name__ == "__main__":
    main()
