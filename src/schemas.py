"""Frozen data contracts shared across all AntiDeepFake-Face modules (M1–M6).

Color & tensor convention (Interface Freeze):
- All images/frames passed between modules are RGB `np.ndarray` with `dtype=np.uint8`.
- `FramePacket.image_rgb` has shape `(H, W, 3)`.
- `FaceCropResult.face_rgb` has shape `(224, 224, 3)` when `has_face=True`, else `None`.
- Canonical final labels: `"Real"`, `"Printed"`, `"On-screen"`, `"Deepfake"`, `"No_Face"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


LABEL_REAL = "Real"
LABEL_PRINTED = "Printed"
LABEL_ONSCREEN = "On-screen"
LABEL_DEEPFAKE = "Deepfake"
LABEL_NO_FACE = "No_Face"

VALID_FINAL_LABELS: Tuple[str, ...] = (
    LABEL_REAL,
    LABEL_PRINTED,
    LABEL_ONSCREEN,
    LABEL_DEEPFAKE,
    LABEL_NO_FACE,
)


@dataclass
class FramePacket:
    """Single decoded frame extracted by M3 (`src/video_io`)."""

    frame_id: int
    timestamp_sec: float
    image_rgb: np.ndarray


@dataclass
class FaceCropResult:
    """Largest detected face crop (224x224x3 RGB uint8) produced by M3 (`src/face_detection`)."""

    frame_id: int
    has_face: bool
    bbox: Optional[Tuple[int, int, int, int]] = None
    det_confidence: float = 0.0
    face_rgb: Optional[np.ndarray] = None
    face_count: int = 0


@dataclass
class SpoofScore:
    """Stage 1 Anti-Spoof probability output produced by M4 (`src/anti_spoof`)."""

    frame_id: int
    probs: Dict[str, float]


@dataclass
class DeepfakeScore:
    """Stage 2 Deepfake probability output produced by M5 (`src/deepfake`)."""

    frame_id: int
    fake_prob: float


@dataclass
class FinalDecision:
    """Aggregated final verdict produced by M1 (`src/decision` & `src/pipeline`)."""

    final_label: str
    confidence: float
    aggregated_probs: Dict[str, float]
    reason: str
    frame_logs: List[dict] = field(default_factory=list)
    preview_frame_rgb: Optional[np.ndarray] = None
    preview_bbox: Optional[Tuple[int, int, int, int]] = None
