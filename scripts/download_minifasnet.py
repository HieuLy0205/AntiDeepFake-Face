"""Download upstream MiniFASNetV2 ONNX weights for Stage 1 Anti-Spoof (M4)."""

from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "models" / "minifasnet_onnx"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="garciafido/minifasnet-v2-anti-spoofing-onnx",
        local_dir=str(OUT_DIR),
    )
    print("Downloaded MiniFASNetV2 ONNX repo to:", path)
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            print(" ", p.relative_to(OUT_DIR), f"({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
