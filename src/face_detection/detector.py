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


_crowd_mtcnn: Optional[object] = None
_crowd_mtcnn_checked: bool = False


def _get_crowd_mtcnn():
    """Khởi tạo và cache MTCNN đa tỉ lệ phục vụ nhận diện đám đông / khuôn mặt nhỏ ở xa."""
    global _crowd_mtcnn, _crowd_mtcnn_checked
    if _crowd_mtcnn_checked:
        return _crowd_mtcnn
    _crowd_mtcnn_checked = True
    try:
        from facenet_pytorch import MTCNN

        _crowd_mtcnn = MTCNN(
            keep_all=True,
            min_face_size=14,
            thresholds=[0.60, 0.70, 0.75],
            factor=0.709,
            device="cpu",
        )
    except Exception:
        _crowd_mtcnn = None
    return _crowd_mtcnn


def _is_valid_face_geometry(
    bbox: Tuple[int, int, int, int],
    landmarks_5x2: Optional[np.ndarray],
) -> bool:
    """Kiểm tra tỷ lệ khung mặt và thứ tự giải phẫu 5 điểm (mắt - mũi - miệng) để loại tai/gáy/vai."""
    x1, y1, x2, y2 = bbox
    bw = float(x2 - x1)
    bh = float(y2 - y1)
    if bw < 12.0 or bh < 12.0:
        return False
    aspect = bw / bh
    if not (0.48 <= aspect <= 1.60):
        return False

    if landmarks_5x2 is None:
        return True
    pts = np.asarray(landmarks_5x2, dtype=np.float32)
    if pts.shape != (5, 2) or not np.all(np.isfinite(pts)):
        return False

    # Tất cả 5 điểm mốc phải nằm hợp lý trong vùng khuôn mặt
    if (
        np.any(pts[:, 0] < x1 - 0.18 * bw)
        or np.any(pts[:, 0] > x2 + 0.18 * bw)
        or np.any(pts[:, 1] < y1 - 0.18 * bh)
        or np.any(pts[:, 1] > y2 + 0.18 * bh)
    ):
        return False

    eye_mid_y = 0.5 * float(pts[0, 1] + pts[1, 1])
    nose_y = float(pts[2, 1])
    mouth_mid_y = 0.5 * float(pts[3, 1] + pts[4, 1])

    # Thứ tự dọc mắt -> mũi -> miệng (đúng cả góc chính diện lẫn nghiêng, loại tai/gáy/cổ)
    if (mouth_mid_y - eye_mid_y) < 0.15 * bh:
        return False
    if max(float(pts[0, 1]), float(pts[1, 1])) > mouth_mid_y + 0.04 * bh:
        return False
    if not (eye_mid_y - 0.08 * bh <= nose_y <= mouth_mid_y + 0.08 * bh):
        return False
    if not (y1 + 0.06 * bh <= eye_mid_y <= y1 + 0.72 * bh):
        return False

    return True


def _nms_face_boxes(
    boxes: List[Tuple[Tuple[int, int, int, int], float]],
    iou_threshold: float = 0.25,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Loại bỏ các khung mặt trùng lặp hoặc lệch nhẹ trên cùng một đầu người."""
    if len(boxes) <= 1:
        return list(boxes)

    sorted_cands = sorted(boxes, key=lambda item: item[1], reverse=True)
    kept: List[Tuple[Tuple[int, int, int, int], float]] = []

    for cand_box, cand_conf in sorted_cands:
        cx1, cy1, cx2, cy2 = cand_box
        cw = max(1, cx2 - cx1)
        ch = max(1, cy2 - cy1)
        c_area = cw * ch
        ccx = cx1 + 0.5 * cw
        ccy = cy1 + 0.5 * ch
        duplicate = False

        for k_box, _k_conf in kept:
            kx1, ky1, kx2, ky2 = k_box
            kw = max(1, kx2 - kx1)
            kh = max(1, ky2 - ky1)
            k_area = kw * kh
            kcx = kx1 + 0.5 * kw
            kcy = ky1 + 0.5 * kh

            ix1 = max(cx1, kx1)
            iy1 = max(cy1, ky1)
            ix2 = min(cx2, kx2)
            iy2 = min(cy2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                iou = inter / float(c_area + k_area - inter)
                contain = inter / float(min(c_area, k_area))
                same_head = (
                    iou >= 0.08
                    and abs(ccx - kcx) < 0.55 * max(cw, kw)
                    and abs(ccy - kcy) < 0.55 * max(ch, kh)
                )
                if iou >= iou_threshold or contain >= 0.55 or same_head:
                    duplicate = True
                    break
        if not duplicate:
            kept.append((cand_box, cand_conf))
    return kept


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

    def _detect_yunet_at_scale(
        self,
        image_rgb: np.ndarray,
        scale: float,
        score_threshold: float,
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Chạy YuNet ở một hệ số phóng đại và ngưỡng tin cậy cho trước, quy đổi tọa độ về ảnh gốc."""
        h, w = image_rgb.shape[:2]
        self.detector.setScoreThreshold(float(score_threshold))
        try:
            if abs(scale - 1.0) > 1e-2:
                up_w = max(1, int(round(w * scale)))
                up_h = max(1, int(round(h * scale)))
                proc_rgb = cv2.resize(image_rgb, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
                inv_scale = 1.0 / float(scale)
            else:
                up_w, up_h = w, h
                proc_rgb = image_rgb
                inv_scale = 1.0

            self.detector.setInputSize((up_w, up_h))
            img_bgr = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR)
            _, faces = self.detector.detect(img_bgr)
            if faces is None or len(faces) == 0:
                return []

            out: List[Tuple[Tuple[int, int, int, int], float]] = []
            for f in faces:
                x, y, bw, bh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                conf = float(f[14])
                x1 = max(0, int(round(x * inv_scale)))
                y1 = max(0, int(round(y * inv_scale)))
                x2 = min(w, int(round((x + bw) * inv_scale)))
                y2 = min(h, int(round((y + bh) * inv_scale)))
                if x2 > x1 and y2 > y1:
                    landmarks = (
                        np.array(
                            [
                                [f[4], f[5]],
                                [f[6], f[7]],
                                [f[8], f[9]],
                                [f[10], f[11]],
                                [f[12], f[13]],
                            ],
                            dtype=np.float32,
                        )
                        * inv_scale
                    )
                    bbox = (x1, y1, x2, y2)
                    if _is_valid_face_geometry(bbox, landmarks):
                        out.append((bbox, conf))
            return out
        finally:
            self.detector.setScoreThreshold(float(self.conf_threshold))
            self.detector.setInputSize((w, h))

    def detect_all_and_crop(
        self,
        packet: FramePacket,
        target_size: Tuple[int, int] = (224, 224),
        scale_ratio: float = 1.2,
        max_faces: int = 20,
        secondary_conf_threshold: float = 0.62,
        min_box_side: int = 16,
    ) -> List[FaceCropResult]:
        """Phát hiện đa tỉ lệ (YuNet + MTCNN) có kiểm tra 5 điểm mốc khuôn mặt và NMS gộp trùng."""
        global _last_face_count
        img = packet.image_rgb
        if img is None or img.ndim != 3:
            self.last_face_count = 0
            _last_face_count = 0
            return []
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            self.last_face_count = 0
            _last_face_count = 0
            return []

        candidates: List[Tuple[Tuple[int, int, int, int], float]] = []
        # Pass 1: YuNet ở độ phân giải gốc
        candidates.extend(
            self._detect_yunet_at_scale(img, scale=1.0, score_threshold=secondary_conf_threshold)
        )

        # Pass 2: YuNet phóng đại cho người ở xa / kích thước mặt nhỏ
        max_dim = max(w, h)
        if max_dim < 1450:
            up_scale = min(2.2, 1500.0 / float(max_dim))
            if up_scale >= 1.25:
                candidates.extend(
                    self._detect_yunet_at_scale(
                        img, scale=up_scale, score_threshold=secondary_conf_threshold
                    )
                )

        # Pass 3: MTCNN kim tự tháp đa tỉ lệ kèm 5 điểm mốc khuôn mặt
        mtcnn = _get_crowd_mtcnn()
        if mtcnn is not None and min(w, h) >= 24:
            try:
                if max_dim < 1250:
                    m_scale = min(2.2, 1300.0 / float(max_dim))
                    m_img = cv2.resize(
                        img,
                        (max(1, int(round(w * m_scale))), max(1, int(round(h * m_scale)))),
                        interpolation=cv2.INTER_CUBIC,
                    )
                else:
                    m_scale = 1.0
                    m_img = img
                mtcnn_min_prob = max(0.86, min(0.98, 0.58 + 0.52 * float(secondary_conf_threshold)))
                m_boxes, m_probs, m_landmarks = mtcnn.detect(m_img, landmarks=True)
                if m_boxes is not None and m_probs is not None:
                    for idx_m, (raw_b, prob) in enumerate(zip(m_boxes, m_probs)):
                        if prob is None or float(prob) < mtcnn_min_prob:
                            continue
                        bx1 = max(0, int(round(float(raw_b[0]) / m_scale)))
                        by1 = max(0, int(round(float(raw_b[1]) / m_scale)))
                        bx2 = min(w, int(round(float(raw_b[2]) / m_scale)))
                        by2 = min(h, int(round(float(raw_b[3]) / m_scale)))
                        if bx2 > bx1 and by2 > by1:
                            pts = None
                            if m_landmarks is not None and idx_m < len(m_landmarks) and m_landmarks[idx_m] is not None:
                                pts = np.asarray(m_landmarks[idx_m], dtype=np.float32) / float(m_scale)
                            bbox = (bx1, by1, bx2, by2)
                            if _is_valid_face_geometry(bbox, pts):
                                candidates.append((bbox, float(prob)))
            except Exception:
                pass

        merged = _nms_face_boxes(candidates, iou_threshold=0.25)
        eff_min_side = min(12, max(8, min(w, h) // 8))
        filtered = [
            (bbox, conf)
            for bbox, conf in merged
            if min(bbox[2] - bbox[0], bbox[3] - bbox[1]) >= eff_min_side
        ]
        sorted_boxes = sorted(
            filtered,
            key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]),
            reverse=True,
        )

        limit = max(1, int(max_faces))
        cropped_items: List[Tuple[Tuple[int, int, int, int], float, np.ndarray]] = []
        for bbox, conf in sorted_boxes[:limit]:
            face_resized = crop_face(
                img,
                bbox,
                target_size=target_size,
                scale_ratio=scale_ratio,
            )
            if face_resized is not None:
                cropped_items.append((bbox, conf, face_resized))

        total_detected = len(cropped_items)
        self.last_face_count = total_detected
        _last_face_count = total_detected
        return [
            FaceCropResult(
                frame_id=packet.frame_id,
                has_face=True,
                bbox=bbox,
                det_confidence=round(float(conf), 4),
                face_rgb=face_resized,
                face_count=total_detected,
            )
            for bbox, conf, face_resized in cropped_items
        ]


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
