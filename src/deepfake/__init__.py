"""Stage 2 Deepfake Detection module (M5) for AntiDeepFake-Face."""

from src.deepfake.predictor import (
    DEFAULT_MODEL_ID,
    DEFAULT_THRESHOLD,
    SUPPORTED_MODELS,
    DeepfakeDetector,
    EnsembleDeepfakeDetector,
    LightweightCNNDetector,
    get_default_detector,
    predict_deepfake,
    predict_deepfake_batch,
    reset_default_detector,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_THRESHOLD",
    "SUPPORTED_MODELS",
    "DeepfakeDetector",
    "EnsembleDeepfakeDetector",
    "LightweightCNNDetector",
    "get_default_detector",
    "predict_deepfake",
    "predict_deepfake_batch",
    "reset_default_detector",
]
