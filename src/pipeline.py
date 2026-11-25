"""End-to-end AntiDeepFake-Face inference pipeline (M1 contract connecting M3, M4, M5, M6).

Flow:
  source (image / video / webcam ID / pre-decoded RGB frames)
    -> [M3] video_io.load_frames()
    -> [M3] face_detection.detect_and_crop()
    -> [M4] anti_spoof.predict_spoof()           (MiniFASNetV2 ONNX + spectral/color Level-2)
    -> [M5] deepfake.predict_deepfake_batch()    (Fine-tuned SigLIP2 / ViT / EfficientNet-B0 / Ensemble)
    -> [M1] decision.aggregate_decision()        (DD6 precedence + DD7 temporal aggregation)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from src.anti_spoof.predictor import get_default_spoof_detector
from src.decision.aggregator import aggregate_decision
from src.deepfake.predictor import (
    DEFAULT_MODEL_ID,
    DeepfakeDetector,
    EnsembleDeepfakeDetector,
    LightweightCNNDetector,
    get_default_detector,
)
from src.face_detection.detector import detect_and_crop
from src.schemas import (
    LABEL_DEEPFAKE,
    LABEL_NO_FACE,
    LABEL_ONSCREEN,
    LABEL_PRINTED,
    LABEL_REAL,
    DeepfakeScore,
    FaceCropResult,
    FinalDecision,
    FramePacket,
    SpoofScore,
)
from src.video_io.reader import load_frames


_CACHED_VARIANTS: Dict[str, Any] = {}


def _get_deepfake_runner(variant: str) -> Any:
    """Cache and return the requested Stage 2 Deepfake detector variant."""
    key = (variant or "siglip2").strip().lower()
    if key == "siglip2":
        return get_default_detector(model_id=DEFAULT_MODEL_ID, device="cpu")
    if key not in _CACHED_VARIANTS:
        if key == "vit":
            _CACHED_VARIANTS[key] = DeepfakeDetector(
                model_id="prithivMLmods/Deep-Fake-Detector-v2-Model",
                device="cpu",
            )
        elif key in ("cnn", "efficientnet_b0"):
            _CACHED_VARIANTS[key] = LightweightCNNDetector(device="cpu")
        elif key == "ensemble":
            det_s = get_default_detector(model_id=DEFAULT_MODEL_ID, device="cpu")
            det_v = _get_deepfake_runner("vit")
            _CACHED_VARIANTS[key] = EnsembleDeepfakeDetector(detectors=[det_s, det_v], weights=[0.5, 0.5])
        else:
            return get_default_detector(model_id=DEFAULT_MODEL_ID, device="cpu")
    return _CACHED_VARIANTS[key]


def _classify_single_frame(
    spoof: SpoofScore,
    df: Optional[DeepfakeScore],
    tau_spoof: float,
    tau_deepfake: float,
) -> str:
    printed = float(spoof.probs.get(LABEL_PRINTED, 0.0))
    onscreen = float(spoof.probs.get(LABEL_ONSCREEN, 0.0))
    if (printed + onscreen) > tau_spoof:
        return LABEL_PRINTED if printed >= onscreen else LABEL_ONSCREEN
    if df is not None and float(df.fake_prob) > tau_deepfake:
        return LABEL_DEEPFAKE
    return LABEL_REAL


def run_pipeline(
    source: Union[str, int, Sequence[np.ndarray]],
    config: Optional[Dict[str, Any]] = None,
) -> FinalDecision:
    """Execute the full AntiDeepFake-Face pipeline on an image, video, webcam ID, or RGB frame list."""
    cfg = config or {}
    sample_fps = int(cfg.get("sample_fps", 5))
    max_frames = int(cfg.get("max_frames", 10))
    enable_deepfake = bool(cfg.get("enable_deepfake", True))
    deepfake_variant = str(cfg.get("deepfake_variant", "siglip2"))

    thr = dict(cfg.get("thresholds") or {})
    tau_spoof = float(thr.get("tau_spoof", thr.get("spoof", 0.60)))
    tau_deepfake = float(thr.get("tau_deepfake", thr.get("deepfake", 0.65)))

    # Step 1: Load RGB uint8 FramePackets (supports pre-buffered webcam RGB arrays as well)
    if isinstance(source, (list, tuple)):
        packets = [
            FramePacket(
                frame_id=idx,
                timestamp_sec=round(idx / max(1, sample_fps), 3),
                image_rgb=np.ascontiguousarray(arr, dtype=np.uint8),
            )
            for idx, arr in enumerate(list(source)[:max_frames])
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[2] == 3
        ]
    else:
        packets = load_frames(source, sample_fps=sample_fps, max_frames=max_frames)

    if not packets:
        return aggregate_decision([], None, thresholds=thr, frame_logs=[])

    spoof_detector = get_default_spoof_detector()

    # Step 2: Face detection + Stage 1 Anti-Spoof per frame
    crops: List[FaceCropResult] = []
    valid_indices: List[int] = []
    spoof_by_idx: Dict[int, SpoofScore] = {}
    pre_df_ms: Dict[int, float] = {}

    for idx, pkt in enumerate(packets):
        t0 = time.perf_counter()
        crop = detect_and_crop(pkt, target_size=(224, 224))
        crops.append(crop)
        if crop.has_face and crop.face_rgb is not None:
            valid_indices.append(idx)
            spoof_by_idx[idx] = spoof_detector.predict(crop, full_frame_rgb=pkt.image_rgb)
        pre_df_ms[idx] = (time.perf_counter() - t0) * 1000.0

    # Step 3: Batched Stage 2 Deepfake inference on valid face crops (for high CPU throughput)
    df_by_idx: Dict[int, DeepfakeScore] = {}
    df_ms_per_valid = 0.0
    if enable_deepfake and valid_indices:
        df_runner = _get_deepfake_runner(deepfake_variant)
        valid_crops = [crops[i] for i in valid_indices]
        t_df0 = time.perf_counter()
        df_list = df_runner.predict_batch(valid_crops)
        df_elapsed_ms = (time.perf_counter() - t_df0) * 1000.0
        df_ms_per_valid = df_elapsed_ms / max(1, len(valid_indices))
        for pos_idx, df_score in zip(valid_indices, df_list):
            df_by_idx[pos_idx] = df_score

    # Step 4: Assemble per-frame timeline logs & choose representative preview frame
    frame_logs: List[dict] = []
    spoof_scores_ordered: List[SpoofScore] = []
    df_scores_ordered: List[DeepfakeScore] = []

    preview_rgb: Optional[np.ndarray] = packets[0].image_rgb
    preview_bbox: Optional[Tuple[int, int, int, int]] = None

    for idx, (pkt, crop) in enumerate(zip(packets, crops)):
        if not crop.has_face or crop.face_rgb is None:
            frame_logs.append(
                {
                    "frame_id": pkt.frame_id,
                    "timestamp_sec": pkt.timestamp_sec,
                    "has_face": False,
                    "face_count": 0,
                    "multi_face": False,
                    "bbox": None,
                    "frame_label": LABEL_NO_FACE,
                    "spoof_probs": {LABEL_REAL: 0.0, LABEL_PRINTED: 0.0, LABEL_ONSCREEN: 0.0},
                    "fake_prob": None,
                    "latency_ms": round(pre_df_ms[idx], 2),
                }
            )
            continue

        if preview_bbox is None and crop.bbox is not None:
            preview_rgb = pkt.image_rgb
            preview_bbox = crop.bbox

        sp_score = spoof_by_idx[idx]
        spoof_scores_ordered.append(sp_score)

        df_score = df_by_idx.get(idx)
        if df_score is not None:
            df_scores_ordered.append(df_score)

        f_label = _classify_single_frame(sp_score, df_score, tau_spoof=tau_spoof, tau_deepfake=tau_deepfake)
        frame_logs.append(
            {
                "frame_id": pkt.frame_id,
                "timestamp_sec": pkt.timestamp_sec,
                "has_face": True,
                "face_count": crop.face_count,
                "multi_face": crop.face_count > 1,
                "bbox": crop.bbox,
                "frame_label": f_label,
                "spoof_probs": dict(sp_score.probs),
                "fake_prob": round(float(df_score.fake_prob), 4) if df_score is not None else None,
                "latency_ms": round(pre_df_ms[idx] + (df_ms_per_valid if enable_deepfake else 0.0), 2),
            }
        )

    # Step 5: Final DD6/DD7 hierarchical decision
    return aggregate_decision(
        spoof_scores=spoof_scores_ordered,
        df_scores=df_scores_ordered if enable_deepfake else None,
        thresholds=thr,
        frame_logs=frame_logs,
        preview_frame_rgb=preview_rgb,
        preview_bbox=preview_bbox,
    )
