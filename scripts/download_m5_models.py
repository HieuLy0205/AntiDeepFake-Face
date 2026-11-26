"""Prefetch candidate HuggingFace Stage 2 Deepfake models into local workspace cache.

Stores model snapshots in `models/hf_cache/` (gitignored) so benchmarking, inference,
and pytest can run 100% offline after initial download.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "models" / "hf_cache"

DEFAULT_MODELS = [
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "prithivMLmods/Deepfake-Detect-Siglip2",
    "dima806/deepfake_vs_real_image_detection",
    "king1oo1/deepfake-model",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pretrained deepfake models to models/hf_cache.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.cache_dir)

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    for model_id in args.models:
        print(f"\n>>> Downloading {model_id} ...")
        try:
            proc = AutoImageProcessor.from_pretrained(model_id, cache_dir=str(args.cache_dir))
            model = AutoModelForImageClassification.from_pretrained(model_id, cache_dir=str(args.cache_dir))
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
            print(
                f"    OK | arch={model.config.model_type} | params={params_m:.1f}M | "
                f"id2label={model.config.id2label} | size={getattr(proc, 'size', None)}"
            )
        except Exception as exc:
            print(f"    SKIP/FAIL ({type(exc).__name__}): {exc}")


if __name__ == "__main__":
    main()
