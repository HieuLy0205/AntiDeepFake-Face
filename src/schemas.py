from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# 1. Đầu ra của video_io (M3)
@dataclass
class FramePacket:
    frame_id: int
    timestamp_sec: float
    image_rgb: np.ndarray  # Bắt buộc: shape (H, W, 3), dtype uint8, hệ màu RGB

# 2. Đầu ra của face_detection (M3) -> Đầu vào chuẩn cho M4 & M5
@dataclass
class FaceCropResult:
    frame_id: int
    has_face: bool
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) trên ảnh gốc
    det_confidence: float = 0.0
    face_rgb: Optional[np.ndarray] = None  # Bắt buộc: shape (224, 224, 3), dtype uint8, hệ RGB

# 3. Đầu ra của anti_spoof (M4)
@dataclass
class SpoofScore:
    frame_id: int
    probs: Dict[str, float]  # Đủ 3 key: {"Real": float, "Printed": float, "On-screen": float}
    latency_ms: float = 0.0

# 4. Đầu ra của deepfake (M5)
@dataclass
class DeepfakeScore:
    frame_id: int
    fake_prob: float  # Xác suất Deepfake trong khoảng [0.0, 1.0]
    latency_ms: float = 0.0

# 5. Đầu ra cuối cùng của decision / pipeline (M1) -> Giao cho M2 & M6
@dataclass
class FinalDecision:
    final_label: str  # 1 trong 5 giá trị: "Real", "Printed", "On-screen", "Deepfake", "No_Face"
    confidence: float  # Điểm tin cậy của nhãn được chọn [0.0, 1.0]
    aggregated_probs: Dict[str, float]  # Xác suất trung bình 4 lớp sau khi gộp N frame
    reason: str  # Giải thích ngắn gọn (VD: "Vuot nguong Printed (0.85 > 0.60)")
    frame_logs: List[dict] = field(default_factory=list)  # Lưu log từng frame cho M6
