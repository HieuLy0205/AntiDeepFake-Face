from pathlib import Path
import pandas as pd

ROOT = Path("data_sampled")
OUT = Path("data/metadata.csv")
LABEL_MAP = {"original_videos": "Real", "inswapper": "Deepfake", "uniface": "Deepfake"}

def main():
    rows = []
    for subdir, label in LABEL_MAP.items():
        folder = ROOT / subdir
        if not folder.exists():
            print(f"[!] Không có {folder}")
            continue
        for i, f in enumerate(sorted(folder.glob("*.mp4"))):
            rows.append({
                "sample_id": f"{subdir}_{f.stem}",
                "filepath": f"{subdir}/{f.name}",
                "label": label,
                "subject_id": f.stem.split("_")[0] if "_" in f.stem else f.stem,
                "split": "val" if i % 5 == 0 else "test",
                "source": "kaggle_ffpp",
                "device": "ffpp",
            })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Đã ghi {len(df)} dòng vào {OUT}")

if __name__ == "__main__":
    main()