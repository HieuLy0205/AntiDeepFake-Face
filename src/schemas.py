from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FramePacket:
    frame_id: int
    timestamp_sec: float
    image_rgb: np.ndarray  # RGB, shape (H, W, 3), dtype uint8


@dataclass
class FaceCropResult:
    frame_id: int
    has_face: bool
    bbox: Optional[Tuple[int, int, int, int]] = None
    det_confidence: float = 0.0
    face_rgb: Optional[np.ndarray] = None  # RGB, shape (224, 224, 3), dtype uint8


@dataclass
class SpoofScore:
    frame_id: int
    probs: Dict[str, float]  # Real, Printed, On-screen; probabilities sum to 1
    latency_ms: float = 0.0


@dataclass
class DeepfakeScore:
    frame_id: int
    fake_prob: float
    latency_ms: float = 0.0


@dataclass
class FinalDecision:
    final_label: str  # Real, Printed, On-screen, Deepfake, or No_Face
    confidence: float
    aggregated_probs: Dict[str, float]
    reason: str
    frame_logs: List[dict] = field(default_factory=list)
