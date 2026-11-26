"""Export leak-free train/val/test MTCNN face crops into a Colab-ready folder & ZIP.

Reads:
  - data/metadata.csv (7 frozen project columns)
  - data/rtfs_m5_rich_metadata.csv (extended metadata for slice analysis)
  - data/cache/rtfs_all_f6/<sample_id>/frame_XX.png (4,800 extracted 224x224 RGB face crops)

Creates:
  - data/cache/rtfs_m5_colab_dataset/
      ├── metadata.csv
      ├── rtfs_m5_rich_metadata.csv
      ├── train/{Real,Deepfake}/<sample_id>__frame_XX.png
      ├── val/{Real,Deepfake}/<sample_id>__frame_XX.png
      └── test/{Real,Deepfake}/<sample_id>__frame_XX.png
  - data/cache/rtfs_m5_colab_dataset.zip  (ready to upload to Google Drive for Colab L4)
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import shutil
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Package train/val/test face crops for Google Colab.")
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data" / "metadata.csv")
    parser.add_argument("--rich-metadata", type=Path, default=REPO_ROOT / "data" / "rtfs_m5_rich_metadata.csv")
    parser.add_argument("--crops-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "rtfs_all_f6")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "rtfs_m5_colab_dataset")
    parser.add_argument("--out-zip", type=Path, default=REPO_ROOT / "data" / "cache" / "rtfs_m5_colab_dataset.zip")
    args = parser.parse_args()

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.metadata, args.out_dir / "metadata.csv")
    shutil.copy2(args.rich_metadata, args.out_dir / "rtfs_m5_rich_metadata.csv")

    with args.rich_metadata.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Verify zero subject_id leakage across splits
    split_subjects = {}
    for r in rows:
        split_subjects.setdefault(r["split"], set()).add(r["subject_id"])
    splits = sorted(split_subjects.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = split_subjects[splits[i]] & split_subjects[splits[j]]
            if overlap:
                raise RuntimeError(f"Leakage between {splits[i]} and {splits[j]}: {len(overlap)} subjects")

    counts = Counter()
    total_images = 0
    for r in rows:
        sid = r["sample_id"]
        split = r["split"]
        label = r["label"]
        src_dir = args.crops_dir / sid
        dst_dir = args.out_dir / split / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        pngs = sorted(src_dir.glob("*.png"))
        if not pngs:
            raise FileNotFoundError(f"Missing extracted crops for {sid} in {src_dir}")
        for png in pngs:
            shutil.copy2(png, dst_dir / f"{sid}__{png.name}")
            counts[(split, label)] += 1
            total_images += 1

    print(f"Exported {total_images} images into {args.out_dir}:")
    for split in ("train", "val", "test"):
        n_real = counts[(split, "Real")]
        n_fake = counts[(split, "Deepfake")]
        print(f"  {split:5s}: Real={n_real:4d}, Deepfake={n_fake:4d}, total={n_real + n_fake:4d}")

    # Create uncompressed/fast-deflate ZIP (PNGs are already compressed -> ZIP_STORED is instant!)
    if args.out_zip.exists():
        args.out_zip.unlink()
    with zipfile.ZipFile(args.out_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for file_path in sorted(args.out_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(args.out_dir.parent).as_posix()
                zf.write(file_path, arcname=arcname)

    zip_mb = args.out_zip.stat().st_size / (1024 * 1024)
    print(f"Created Colab ZIP: {args.out_zip} ({zip_mb:.1f} MB)")


if __name__ == "__main__":
    main()
