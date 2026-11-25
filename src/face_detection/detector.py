import os
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.schemas import FramePacket, FaceCropResult

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "face_detection_yunet_2023mar.onnx",
)

# Theo dõi số lượng khuôn mặt ở lần gọi gần nhất (phục vụ log nhiều khuôn mặt cho M1/M6)
_last_face_count: int = 0


def get_last_face_count() -> int:
    """Trả về số khuôn mặt phát hiện được ở lần chạy `detect_and_crop` gần nhất."""
    return _last_face_count


def crop_face(
    image_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = (224, 224),
    scale_ratio: float = 1.2,
) -> Optional[np.ndarray]:
    """
    Cắt khuôn mặt theo hình vuông (Square Crop) mở rộng theo `scale_ratio`,
    clamp an toàn vào biên ảnh và resize về `target_size` (RGB uint8).
    Hỗ trợ crop đa tỉ lệ cho từng model downstream (VD: M4 MiniFASNet vs M5 Deepfake).
    """
    if image_rgb is None or image_rgb.ndim != 3:
        return None
    h, w, _ = image_rgb.shape
    if h <= 0 or w <= 0:
        return None

    x1_raw, y1_raw, x2_raw, y2_raw = bbox
    box_w = float(x2_raw - x1_raw)
    box_h = float(y2_raw - y1_raw)
    if box_w <= 0 or box_h <= 0:
        return None

    cx = x1_raw + box_w / 2.0
    cy = y1_raw + box_h / 2.0
    max_side = max(box_w, box_h) * float(scale_ratio)

    sx1 = max(0, int(round(cx - max_side / 2.0)))
    sy1 = max(0, int(round(cy - max_side / 2.0)))
    sx2 = min(w, int(round(cx + max_side / 2.0)))
    sy2 = min(h, int(round(cy + max_side / 2.0)))

    crop = image_rgb[sy1:sy2, sx1:sx2]
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return None

    face_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    return face_resized.astype(np.uint8)


class FaceDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        """Khởi tạo bộ phát hiện khuôn mặt YuNet dựa trên OpenCV DNN."""
        self.model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(self.model_path):
            from scripts.download_yunet import download_model

            self.model_path = download_model()

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.last_face_count: int = 0

        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )

    def detect_raw_boxes(
        self, image_rgb: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Phát hiện toàn bộ khuôn mặt trên ảnh RGB uint8.
        Trả về danh sách [((x1, y1, x2, y2), confidence), ...].
        """
        if image_rgb is None or image_rgb.ndim != 3:
            return []
        h, w, _ = image_rgb.shape
        if h <= 0 or w <= 0:
            return []

        self.detector.setInputSize((w, h))
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, faces = self.detector.detect(img_bgr)

        if faces is None or len(faces) == 0:
            return []

        boxes: List[Tuple[Tuple[int, int, int, int], float]] = []
        for f in faces:
            x, y, bw, bh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
            conf = float(f[14])
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(w, int(round(x + bw)))
            y2 = min(h, int(round(y + bh)))
            if x2 > x1 and y2 > y1:
                boxes.append(((x1, y1, x2, y2), conf))
        return boxes

    def detect_and_crop(
        self,
        packet: FramePacket,
        target_size: Tuple[int, int] = (224, 224),
        scale_ratio: float = 1.2,
    ) -> FaceCropResult:
        """
        Phát hiện khuôn mặt trên FramePacket, chọn mặt lớn nhất,
        crop hình vuông theo `scale_ratio` và resize về `target_size` (RGB uint8).
        """
        boxes = self.detect_raw_boxes(packet.image_rgb)
        self.last_face_count = len(boxes)
        return _build_face_crop_result(
            packet=packet,
            boxes=boxes,
            target_size=target_size,
            scale_ratio=scale_ratio,
        )


_global_detector: Optional[FaceDetector] = None


def _get_global_detector() -> FaceDetector:
    global _global_detector
    if _global_detector is None:
        _global_detector = FaceDetector()
    return _global_detector


def _detect_raw_boxes(
    image_rgb: np.ndarray,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Helper cấp module trả về danh sách ((x1, y1, x2, y2), conf) để hỗ trợ unit test mock."""
    return _get_global_detector().detect_raw_boxes(image_rgb)


def _build_face_crop_result(
    packet: FramePacket,
    boxes: List[Tuple[Tuple[int, int, int, int], float]],
    target_size: Tuple[int, int],
    scale_ratio: float,
) -> FaceCropResult:
    global _last_face_count
    _last_face_count = len(boxes)

    if not boxes:
        return FaceCropResult(
            frame_id=packet.frame_id,
            has_face=False,
            bbox=None,
            det_confidence=0.0,
            face_rgb=None,
        )

    best_box, best_conf = max(
        boxes,
        key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]),
    )

    face_resized = crop_face(
        packet.image_rgb,
        best_box,
        target_size=target_size,
        scale_ratio=scale_ratio,
    )
    if face_resized is None:
        return FaceCropResult(
            frame_id=packet.frame_id,
            has_face=False,
            bbox=None,
            det_confidence=0.0,
            face_rgb=None,
        )

    return FaceCropResult(
        frame_id=packet.frame_id,
        has_face=True,
        bbox=best_box,
        det_confidence=round(float(best_conf), 4),
        face_rgb=face_resized,
    )


def detect_and_crop(
    packet: FramePacket,
    target_size: Tuple[int, int] = (224, 224),
    scale_ratio: float = 1.2,
) -> FaceCropResult:
    """
    Hàm module-level theo đúng chữ ký của Quy ước Design:
    def detect_and_crop(packet: FramePacket, target_size: Tuple[int, int] = (224, 224)) -> FaceCropResult
    """
    boxes = _detect_raw_boxes(packet.image_rgb)
    return _build_face_crop_result(
        packet=packet,
        boxes=boxes,
        target_size=target_size,
        scale_ratio=scale_ratio,
    )
