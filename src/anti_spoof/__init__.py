"""Stage 1 Face Anti-Spoofing module (M4 contract)."""

from src.anti_spoof.predictor import AntiSpoofDetector, get_default_spoof_detector, predict_spoof

__all__ = ["AntiSpoofDetector", "get_default_spoof_detector", "predict_spoof"]
