"""Stage 2 Deepfake Detector wrapper (M5) supporting PyTorch (HuggingFace) & ONNX CPU.

Conforms to the frozen AntiDeepFake-Face interface:
- Input:  `FaceCropResult` (`face_rgb` shape `(224, 224, 3)`, RGB `np.uint8`)
- Output: `DeepfakeScore(frame_id=..., fake_prob=...)` where `0.0 <= fake_prob <= 1.0`
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.schemas import DeepfakeScore, FaceCropResult
from src.deepfake.preprocess import (
    DEFAULT_IMAGE_SIZE,
    HALF_MEAN,
    HALF_STD,
    face_rgb_to_pil,
    preprocess_for_onnx,
    validate_rgb_face,
)


# Curated HuggingFace pretrained models benchmarked on dataset_RTFS_800_Subset
SUPPORTED_MODELS: Dict[str, Dict[str, str]] = {
    "prithivMLmods/Deep-Fake-Detector-v2-Model": {
        "backbone": "google/vit-base-patch16-224-in21k",
        "family": "ViT-Base",
        "notes": "Fine-tuned on Deepfake-vs-Real-v2 (Realism vs Deepfake), 224x224.",
    },
    "prithivMLmods/Deepfake-Detect-Siglip2": {
        "backbone": "google/siglip2-base-patch16-224",
        "family": "SigLIP2-Base",
        "notes": "Vision-language SigLIP2 backbone fine-tuned for real vs fake faces, 224x224.",
    },
    "dima806/deepfake_vs_real_image_detection": {
        "backbone": "google/vit-base-patch16-224-in21k",
        "family": "ViT-Base",
        "notes": "Fine-tuned on 190k real vs deepfake faces, 224x224.",
    },
    "king1oo1/deepfake-model": {
        "backbone": "google/siglip2-base-patch16-224",
        "family": "SigLIP2-Base",
        "notes": "Fine-tuned on combined FaceForensics++, DFDC, and Celeb-DF, 224x224.",
    },
}

# Default model selected empirically from RTFS-800 held-out test E5 benchmark:
# king1oo1/deepfake-model + m5_rtfs_siglip2_adapter.pt achieved TEST ROC-AUC = 0.9656,
# EER = 0.0667, ACER = 0.0900 at val-calibrated tau_d = 0.6693 (11-14 FPS on CPU).
DEFAULT_MODEL_ID = "king1oo1/deepfake-model"
DEFAULT_THRESHOLD = 0.6693
DEFAULT_HF_CACHE_DIR = Path(__file__).resolve().parents[2] / "models" / "hf_cache"
DEFAULT_ADAPTER_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "models" / "m5_rtfs_siglip2_adapter.pt",
    Path(__file__).resolve().parents[2] / "chẹckpoint" / "m5_rtfs_siglip2_adapter.pt",
    Path(__file__).resolve().parents[2] / "models" / "m5_rtfs_siglip2_top2.pt",
)
DEFAULT_VIT_ADAPTER_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "models" / "m5_rtfs_vit_adapter.pt",
    Path(__file__).resolve().parents[2] / "chẹckpoint" / "m5_rtfs_vit_adapter.pt",
)
DEFAULT_CNN_CHECKPOINT_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "models" / "m5_rtfs_efficientnet_b0.pt",
    Path(__file__).resolve().parents[2] / "chẹckpoint" / "m5_rtfs_efficientnet_b0.pt",
)
DEFAULT_ADAPTER_PATH = DEFAULT_ADAPTER_CANDIDATES[0]

_FAKE_TOKENS = {
    "fake",
    "deepfake",
    "deepfakes",
    "ai",
    "generated",
    "synthetic",
    "manipulated",
    "forged",
    "spoof",
    "label_1_fake",
}
_REAL_TOKENS = {
    "real",
    "realism",
    "authentic",
    "genuine",
    "human",
    "original",
    "bonafide",
    "bona_fide",
}


def resolve_fake_label_index(id2label: Dict[Any, str]) -> int:
    """Safely determine which logit index corresponds to the Fake/Deepfake class.

    Prevents silent label inversion across HuggingFace checkpoints whose `id2label`
    mappings vary (e.g., `{0: 'Realism', 1: 'Deepfake'}` vs `{0: 'Fake', 1: 'Real'}`).
    """
    if not id2label:
        raise ValueError("Model config is missing id2label mapping.")

    normalized: Dict[int, str] = {int(k): str(v).strip().lower() for k, v in id2label.items()}

    fake_indices = [
        idx
        for idx, name in normalized.items()
        if any(tok in name for tok in _FAKE_TOKENS)
    ]
    if len(fake_indices) == 1:
        return fake_indices[0]

    # Fallback: if binary model and real label is unambiguous, take the other index
    if len(normalized) == 2:
        real_indices = [
            idx
            for idx, name in normalized.items()
            if any(tok in name for tok in _REAL_TOKENS)
        ]
        if len(real_indices) == 1:
            other = [idx for idx in normalized.keys() if idx != real_indices[0]]
            return other[0]

    raise ValueError(
        f"Cannot unambiguously resolve Fake class index from id2label={id2label}. "
        "Please pass fake_label_index explicitly."
    )


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


class DeepfakeDetector:
    """Wrapper for Stage 2 Deepfake inference on CPU (PyTorch or ONNX Runtime)."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cpu",
        backend: str = "pytorch",
        onnx_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        use_default_adapter: bool = True,
        fake_label_index: Optional[int] = None,
        local_files_only: bool = False,
        num_threads: Optional[int] = None,
    ) -> None:
        self.model_id = model_id or os.environ.get("ADF_DEEPFAKE_MODEL_ID", DEFAULT_MODEL_ID)
        self.device = device
        self.onnx_path = onnx_path or os.environ.get("ADF_DEEPFAKE_ONNX_PATH")
        self.backend = "onnx" if (backend == "onnx" or self.onnx_path) else "pytorch"
        self.local_files_only = local_files_only
        self.num_threads = num_threads

        env_adapter = os.environ.get("ADF_DEEPFAKE_ADAPTER_PATH")
        if adapter_path is not None:
            self.adapter_path: Optional[str] = str(adapter_path) if adapter_path else None
        elif env_adapter is not None:
            self.adapter_path = env_adapter or None
        elif (
            use_default_adapter
            and self.backend == "pytorch"
        ):
            candidates = (
                DEFAULT_ADAPTER_CANDIDATES
                if self.model_id == DEFAULT_MODEL_ID
                else (
                    DEFAULT_VIT_ADAPTER_CANDIDATES
                    if self.model_id == "prithivMLmods/Deep-Fake-Detector-v2-Model"
                    else ()
                )
            )
            found_adapter = next((p for p in candidates if p.is_file()), None)
            self.adapter_path = str(found_adapter) if found_adapter is not None else None
        else:
            self.adapter_path = None
        self.adapter_loaded: bool = False
        self.recommended_threshold: float = DEFAULT_THRESHOLD

        self._model: Any = None
        self._processor: Any = None
        self._ort_session: Any = None
        self._ort_input_name: Optional[str] = None

        self.id2label: Dict[int, str] = {}
        self.fake_label_index: Optional[int] = fake_label_index
        self.image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
        self.image_mean: Tuple[float, float, float] = HALF_MEAN
        self.image_std: Tuple[float, float, float] = HALF_STD

        if self.backend == "onnx":
            self._load_onnx()
        else:
            self._load_pytorch()

    def _load_pytorch(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        if self.num_threads is not None and self.num_threads > 0:
            torch.set_num_threads(int(self.num_threads))

        cache_dir = os.environ.get("ADF_HF_CACHE_DIR")
        if cache_dir is None and DEFAULT_HF_CACHE_DIR.is_dir():
            cache_dir = str(DEFAULT_HF_CACHE_DIR)

        load_kwargs: Dict[str, Any] = {}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        if self.local_files_only:
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_id, local_files_only=True, **load_kwargs
            )
            self._model = AutoModelForImageClassification.from_pretrained(
                self.model_id, local_files_only=True, **load_kwargs
            )
        else:
            # Try instant local snapshot first to avoid network latency; fall back to Hub download
            try:
                self._processor = AutoImageProcessor.from_pretrained(
                    self.model_id, local_files_only=True, **load_kwargs
                )
                self._model = AutoModelForImageClassification.from_pretrained(
                    self.model_id, local_files_only=True, **load_kwargs
                )
            except Exception:
                self._processor = AutoImageProcessor.from_pretrained(
                    self.model_id, local_files_only=False, **load_kwargs
                )
                self._model = AutoModelForImageClassification.from_pretrained(
                    self.model_id, local_files_only=False, **load_kwargs
                )
        self._model.eval()
        self._model.to(self.device)

        if self.adapter_path:
            self._apply_adapter(Path(self.adapter_path))

        raw_id2label = getattr(self._model.config, "id2label", {}) or {}
        self.id2label = {int(k): str(v) for k, v in raw_id2label.items()}
        if self.fake_label_index is None:
            self.fake_label_index = resolve_fake_label_index(self.id2label)

        # Extract processor geometry & normalization for ONNX export parity (handles both dict & v5 SizeDict)
        size_cfg = getattr(self._processor, "size", None)
        if isinstance(size_cfg, dict):
            h = int(size_cfg.get("height") or size_cfg.get("shortest_edge") or 224)
            w = int(size_cfg.get("width") or size_cfg.get("shortest_edge") or h)
            self.image_size = (w, h)
        elif hasattr(size_cfg, "height") or hasattr(size_cfg, "shortest_edge"):
            h = int(getattr(size_cfg, "height", None) or getattr(size_cfg, "shortest_edge", None) or 224)
            w = int(getattr(size_cfg, "width", None) or getattr(size_cfg, "shortest_edge", None) or h)
            self.image_size = (w, h)
        elif isinstance(size_cfg, (int, float)):
            self.image_size = (int(size_cfg), int(size_cfg))

        mean_cfg = getattr(self._processor, "image_mean", None)
        std_cfg = getattr(self._processor, "image_std", None)
        if mean_cfg and len(mean_cfg) == 3:
            self.image_mean = (float(mean_cfg[0]), float(mean_cfg[1]), float(mean_cfg[2]))
        if std_cfg and len(std_cfg) == 3:
            self.image_std = (float(std_cfg[0]), float(std_cfg[1]), float(std_cfg[2]))

    def _apply_adapter(self, adapter_file: Path) -> None:
        """Apply a saved M5 partial fine-tuned adapter state dict onto the loaded HF model."""
        import torch

        if not adapter_file.is_file():
            raise FileNotFoundError(f"M5 adapter checkpoint not found: {adapter_file}")
        payload = torch.load(str(adapter_file), map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError(f"Invalid M5 adapter checkpoint (missing state_dict): {adapter_file}")

        expected_model = payload.get("model_id")
        if expected_model and expected_model != self.model_id:
            raise ValueError(
                f"Adapter {adapter_file.name} was trained for '{expected_model}', "
                f"cannot apply to '{self.model_id}'."
            )

        target_dtype = next(self._model.parameters()).dtype
        cast_sd = {k: v.to(device=self.device, dtype=target_dtype) for k, v in state_dict.items()}
        _missing, unexpected = self._model.load_state_dict(cast_sd, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected parameter keys in adapter {adapter_file}: {unexpected[:5]}")

        if "recommended_threshold" in payload:
            self.recommended_threshold = float(payload["recommended_threshold"])
        self.adapter_loaded = True

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        if not self.onnx_path or not Path(self.onnx_path).is_file():
            raise FileNotFoundError(f"ONNX model file not found: {self.onnx_path}")

        onnx_file = Path(self.onnx_path)
        meta_file = onnx_file.with_suffix(".meta.json")
        if meta_file.is_file():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            self.model_id = meta.get("model_id", self.model_id)
            self.id2label = {int(k): str(v) for k, v in (meta.get("id2label") or {}).items()}
            if self.fake_label_index is None and "fake_label_index" in meta:
                self.fake_label_index = int(meta["fake_label_index"])
            if "image_size" in meta:
                self.image_size = (int(meta["image_size"][0]), int(meta["image_size"][1]))
            if "image_mean" in meta:
                self.image_mean = tuple(float(x) for x in meta["image_mean"])  # type: ignore[assignment]
            if "image_std" in meta:
                self.image_std = tuple(float(x) for x in meta["image_std"])  # type: ignore[assignment]

        if self.fake_label_index is None:
            if self.id2label:
                self.fake_label_index = resolve_fake_label_index(self.id2label)
            else:
                raise ValueError(
                    "ONNX metadata missing `fake_label_index`. Provide `.meta.json` or `fake_label_index`."
                )

        sess_opts = ort.SessionOptions()
        if self.num_threads is not None and self.num_threads > 0:
            sess_opts.intra_op_num_threads = int(self.num_threads)
        self._ort_session = ort.InferenceSession(
            str(onnx_file),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._ort_input_name = self._ort_session.get_inputs()[0].name

    def predict_probs_raw(self, faces_rgb: Sequence[np.ndarray]) -> List[float]:
        """Run batched inference on raw RGB uint8 face crops and return fake probabilities."""
        if not faces_rgb:
            return []

        assert self.fake_label_index is not None

        if self.backend == "onnx":
            pixel_values = preprocess_for_onnx(
                faces_rgb,
                image_size=self.image_size,
                mean=self.image_mean,
                std=self.image_std,
            )
            ort_outs = self._ort_session.run(None, {self._ort_input_name: pixel_values})
            probs = _softmax_np(np.asarray(ort_outs[0], dtype=np.float32))
            return [float(np.clip(p, 0.0, 1.0)) for p in probs[:, self.fake_label_index]]

        import torch

        pil_images = [face_rgb_to_pil(validate_rgb_face(f)) for f in faces_rgb]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, self.fake_label_index]
        return [float(np.clip(v, 0.0, 1.0)) for v in probs.detach().cpu().tolist()]

    def predict(self, face_crop: FaceCropResult) -> DeepfakeScore:
        """Predict deepfake probability for a single `FaceCropResult`."""
        if not face_crop.has_face or face_crop.face_rgb is None:
            return DeepfakeScore(frame_id=int(face_crop.frame_id), fake_prob=0.0)
        fake_prob = self.predict_probs_raw([face_crop.face_rgb])[0]
        return DeepfakeScore(frame_id=int(face_crop.frame_id), fake_prob=fake_prob)

    def predict_batch(self, face_crops: Sequence[FaceCropResult]) -> List[DeepfakeScore]:
        """Predict deepfake probabilities for a sequence of frames in one batched pass."""
        if not face_crops:
            return []

        valid_positions: List[int] = []
        valid_faces: List[np.ndarray] = []
        for idx, crop in enumerate(face_crops):
            if crop.has_face and crop.face_rgb is not None:
                valid_positions.append(idx)
                valid_faces.append(crop.face_rgb)

        scores: List[DeepfakeScore] = [
            DeepfakeScore(frame_id=int(c.frame_id), fake_prob=0.0) for c in face_crops
        ]
        if valid_faces:
            probs = self.predict_probs_raw(valid_faces)
            for pos, prob in zip(valid_positions, probs):
                scores[pos] = DeepfakeScore(frame_id=int(face_crops[pos].frame_id), fake_prob=prob)
        return scores

    def export_to_onnx(self, output_path: str | Path, quantize_int8: bool = False) -> Path:
        """Export the loaded PyTorch model to ONNX (+ optional dynamic INT8 quantization)."""
        if self.backend != "pytorch" or self._model is None:
            raise RuntimeError("export_to_onnx requires a PyTorch-loaded DeepfakeDetector.")

        import torch

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        w, h = self.image_size
        dummy = torch.randn(1, 3, h, w, dtype=torch.float32, device="cpu")

        model_cpu = self._model.to("cpu").eval()

        class _LogitsWrapper(torch.nn.Module):
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                return self.inner(pixel_values=pixel_values).logits

        wrapper = _LogitsWrapper(model_cpu)
        fp32_path = out_path if not quantize_int8 else out_path.with_suffix(".fp32.onnx")
        torch.onnx.export(
            wrapper,
            dummy,
            str(fp32_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )

        final_path = fp32_path
        if quantize_int8:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                model_input=str(fp32_path),
                model_output=str(out_path),
                weight_type=QuantType.QUInt8,
            )
            final_path = out_path

        meta = {
            "model_id": self.model_id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "fake_label_index": int(self.fake_label_index) if self.fake_label_index is not None else 1,
            "image_size": [w, h],
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
            "quantized_int8": bool(quantize_int8),
        }
        final_path.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return final_path


_DEFAULT_DETECTOR: Optional[DeepfakeDetector] = None


def get_default_detector(
    model_id: Optional[str] = None,
    device: str = "cpu",
    backend: str = "pytorch",
    onnx_path: Optional[str] = None,
) -> DeepfakeDetector:
    """Return a lazily-initialized singleton DeepfakeDetector (cached in RAM per DD8)."""
    global _DEFAULT_DETECTOR
    requested_model = model_id or os.environ.get("ADF_DEEPFAKE_MODEL_ID", DEFAULT_MODEL_ID)
    requested_onnx = onnx_path or os.environ.get("ADF_DEEPFAKE_ONNX_PATH")
    requested_backend = "onnx" if (backend == "onnx" or requested_onnx) else "pytorch"

    if (
        _DEFAULT_DETECTOR is None
        or _DEFAULT_DETECTOR.model_id != requested_model
        or _DEFAULT_DETECTOR.backend != requested_backend
        or _DEFAULT_DETECTOR.onnx_path != requested_onnx
    ):
        _DEFAULT_DETECTOR = DeepfakeDetector(
            model_id=requested_model,
            device=device,
            backend=requested_backend,
            onnx_path=requested_onnx,
        )
    return _DEFAULT_DETECTOR


def reset_default_detector() -> None:
    """Clear the cached singleton detector (useful for tests or switching models)."""
    global _DEFAULT_DETECTOR
    _DEFAULT_DETECTOR = None


def predict_deepfake(
    face_crop: FaceCropResult,
    detector: Optional[DeepfakeDetector] = None,
) -> DeepfakeScore:
    """Canonical M5 entry point invoked by M1 (`src/pipeline.py`) and `tests/test_anti_spoof_deepfake.py`."""
    if not face_crop.has_face or face_crop.face_rgb is None:
        return DeepfakeScore(frame_id=int(face_crop.frame_id), fake_prob=0.0)
    det = detector if detector is not None else get_default_detector()
    return det.predict(face_crop)


def predict_deepfake_batch(
    face_crops: Sequence[FaceCropResult],
    detector: Optional[DeepfakeDetector] = None,
) -> List[DeepfakeScore]:
    """Batched M5 entry point for multi-frame videos (N=5..30) on CPU."""
    if not face_crops:
        return []
    det = detector if detector is not None else get_default_detector()
    return det.predict_batch(face_crops)


class EnsembleDeepfakeDetector:
    """Weighted probability ensemble combining complementary Stage 2 Deepfake detectors.

    Default configuration pairs `king1oo1/deepfake-model` (strong on UniFace) with
    `prithivMLmods/Deep-Fake-Detector-v2-Model` (strong on InSwapper), raising overall
    validation ROC-AUC on RTFS-800 from 0.6203 to 0.6610 while staying ~8 FPS on CPU.
    """

    def __init__(
        self,
        detectors: Optional[Sequence[DeepfakeDetector]] = None,
        weights: Optional[Sequence[float]] = None,
        device: str = "cpu",
    ) -> None:
        if detectors is None:
            self.detectors = [
                DeepfakeDetector("king1oo1/deepfake-model", device=device),
                DeepfakeDetector("prithivMLmods/Deep-Fake-Detector-v2-Model", device=device),
            ]
        else:
            self.detectors = list(detectors)

        if not self.detectors:
            raise ValueError("EnsembleDeepfakeDetector requires at least one detector.")

        if weights is None:
            raw_w = [1.0 / len(self.detectors)] * len(self.detectors)
        else:
            if len(weights) != len(self.detectors):
                raise ValueError("weights length must match detectors length.")
            total = float(sum(weights))
            if total <= 0:
                raise ValueError("Sum of ensemble weights must be > 0.")
            raw_w = [float(w) / total for w in weights]
        self.weights = raw_w

    def predict(self, face_crop: FaceCropResult) -> DeepfakeScore:
        if not face_crop.has_face or face_crop.face_rgb is None:
            return DeepfakeScore(frame_id=int(face_crop.frame_id), fake_prob=0.0)
        score = sum(w * det.predict(face_crop).fake_prob for w, det in zip(self.weights, self.detectors))
        return DeepfakeScore(frame_id=int(face_crop.frame_id), fake_prob=float(np.clip(score, 0.0, 1.0)))

    def predict_batch(self, face_crops: Sequence[FaceCropResult]) -> List[DeepfakeScore]:
        if not face_crops:
            return []
        per_det = [det.predict_batch(face_crops) for det in self.detectors]
        combined: List[DeepfakeScore] = []
        for idx, crop in enumerate(face_crops):
            if not crop.has_face or crop.face_rgb is None:
                combined.append(DeepfakeScore(frame_id=int(crop.frame_id), fake_prob=0.0))
                continue
            prob = sum(w * scores[idx].fake_prob for w, scores in zip(self.weights, per_det))
            combined.append(DeepfakeScore(frame_id=int(crop.frame_id), fake_prob=float(np.clip(prob, 0.0, 1.0))))
        return combined


class LightweightCNNDetector:
    """Fast CPU Stage 2 Deepfake detector backed by fine-tuned `timm/efficientnet_b0` (4.0M params).

    Achieved TEST ROC-AUC = 0.9529, BPCER = 0.0200 on RTFS-800 while running >30 FPS on laptop CPU.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: str = "cpu",
    ) -> None:
        import timm
        import torch

        if checkpoint_path is not None:
            ckpt_file = Path(checkpoint_path)
        else:
            found = next((p for p in DEFAULT_CNN_CHECKPOINT_CANDIDATES if p.is_file()), None)
            if found is None:
                raise FileNotFoundError(
                    "Lightweight CNN checkpoint m5_rtfs_efficientnet_b0.pt not found in models/."
                )
            ckpt_file = found

        if not ckpt_file.is_file():
            raise FileNotFoundError(f"CNN checkpoint not found: {ckpt_file}")

        payload = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        self.arch = str(payload.get("arch", "efficientnet_b0"))
        self.checkpoint_path = str(ckpt_file)
        self.device = device
        self.recommended_threshold = float(payload.get("recommended_threshold", 0.5861))
        self.image_size = DEFAULT_IMAGE_SIZE
        self.image_mean = (0.485, 0.456, 0.406)
        self.image_std = (0.229, 0.224, 0.225)

        self._model = timm.create_model(self.arch, pretrained=False, num_classes=2)
        self._model.load_state_dict(payload["state_dict"], strict=True)
        self._model.eval()
        self._model.to(self.device)

    def predict_probs_raw(self, faces_rgb: Sequence[np.ndarray]) -> List[float]:
        if not faces_rgb:
            return []
        import torch

        batch_np = preprocess_for_onnx(
            faces_rgb,
            image_size=self.image_size,
            mean=self.image_mean,
            std=self.image_std,
        )
        tensor = torch.from_numpy(batch_np).to(self.device)
        with torch.inference_mode():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=-1)[:, 1]
        return [float(np.clip(v, 0.0, 1.0)) for v in probs.detach().cpu().tolist()]

    def predict(self, face_crop: FaceCropResult) -> DeepfakeScore:
        if not face_crop.has_face or face_crop.face_rgb is None:
            return DeepfakeScore(frame_id=int(face_crop.frame_id), fake_prob=0.0)
        return DeepfakeScore(
            frame_id=int(face_crop.frame_id),
            fake_prob=self.predict_probs_raw([face_crop.face_rgb])[0],
        )

    def predict_batch(self, face_crops: Sequence[FaceCropResult]) -> List[DeepfakeScore]:
        if not face_crops:
            return []
        valid_pos = [i for i, c in enumerate(face_crops) if c.has_face and c.face_rgb is not None]
        valid_faces = [face_crops[i].face_rgb for i in valid_pos]  # type: ignore[misc]
        scores = [DeepfakeScore(frame_id=int(c.frame_id), fake_prob=0.0) for c in face_crops]
        if valid_faces:
            probs = self.predict_probs_raw(valid_faces)  # type: ignore[arg-type]
            for pos, prob in zip(valid_pos, probs):
                scores[pos] = DeepfakeScore(frame_id=int(face_crops[pos].frame_id), fake_prob=prob)
        return scores
