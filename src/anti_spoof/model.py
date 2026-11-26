"""Bộ ghép xác suất M4 cho ba lớp real, printed và onscreen.

Đầu vào là ảnh crop RGB uint8 (H x W x 3) từ M3.
Đầu ra là dict có ba xác suất chữ thường, tổng bằng 1.
Level 1 dùng MiniFASNet; Level 2 tạm dùng mock cho tới khi nhận code 4.2.
"""

from collections.abc import Callable
import math

import numpy as np

# Level 1 trả về (xác suất thật, xác suất giả); Level 2 trả về
# (xác suất ảnh in, xác suất màn hình) trong nhóm ảnh giả.
Level1Predictor = Callable[[np.ndarray], tuple[float, float]]
Level2Predictor = Callable[[np.ndarray], tuple[float, float]]

def _predict_level1(crop_rgb: np.ndarray) -> tuple[float, float]:
    """Gọi MiniFASNet thật; model và weights được nạp/cache tại level1.py."""
    from .level1 import predict_level1

    return predict_level1(crop_rgb)

def _mock_level2(crop_rgb: np.ndarray) -> tuple[float, float]:
    """Mock tạm để kiểm tra luồng; sẽ thay bằng heuristic OpenCV của 4.2."""
    return 0.40, 0.60

def _normalize_pair(values: tuple[float, float], name: str) -> tuple[float, float]:
    """Kiểm tra một cặp điểm và chuẩn hóa để tổng của cặp bằng 1."""
    try:
        first, second = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must return exactly two probabilities.") from exc

    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError(f"{name} probabilities must be finite.")
    if not 0.0 <= first <= 1.0 or not 0.0 <= second <= 1.0:
        raise ValueError(f"{name} probabilities must be between 0 and 1.")

    total = first + second
    if total <= 0.0:
        raise ValueError(f"{name} probabilities must have a positive sum.")
    return first / total, second / total

class AntiSpoofModel:
    """Ghép kết quả Level 1 và Level 2 thành xác suất ba lớp.
    Đây là cách phân loại phân cấp: Level 1 tách thật/giả; nếu đủ ngưỡng giả,
    Level 2 chia phần giả thành ảnh in và ảnh trên màn hình.
    """
    def __init__(
        self,
        spoof_threshold: float = 0.5,
        *,
        level1_predictor: Level1Predictor | None = None,
        level2_predictor: Level2Predictor | None = None,
    ) -> None:
        """Thiết lập ngưỡng và hàm dự đoán; cho phép tiêm mock khi viết test.
        Có thể truyền predictor khác khi test; mặc định dùng MiniFASNet cho Level 1
        và mock cho Level 2 tới khi phần heuristic của 4.2 được bàn giao.
        """
        threshold = float(spoof_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("spoof_threshold must be between 0 and 1.")

        self.spoof_threshold = threshold
        self._level1_predictor = (
            level1_predictor if level1_predictor is not None else _predict_level1
        )
        self._level2_predictor = (
            level2_predictor if level2_predictor is not None else _mock_level2
        )
    def predict(self, crop: np.ndarray | None) -> dict[str, float]:
        """Nhận crop RGB từ M3, trả xác suất real/printed/onscreen.
        Ảnh rỗng, sai số kênh hoặc sai dtype bị từ chối, không giả làm kết quả
        xác suất hợp lệ bằng cách trả ba số 0.
        """
        if (
            not isinstance(crop, np.ndarray)
            or crop.ndim != 3
            or crop.shape[0] == 0
            or crop.shape[1] == 0
            or crop.shape[2] != 3
            or crop.dtype != np.uint8
        ):
            raise ValueError("crop must be a non-empty RGB uint8 array with shape HxWx3.")

        # Level 1: phân biệt khuôn mặt thật với ảnh giả.
        p_real, p_spoof = _normalize_pair(
            self._level1_predictor(crop), "Level 1"
        )

        # Level 2 chỉ phân loại tiếp khi điểm giả đạt ngưỡng.
        if p_spoof >= self.spoof_threshold:
            w_printed, w_onscreen = _normalize_pair(
                self._level2_predictor(crop), "Level 2"
            )
        else:
            # Chưa đủ căn cứ gọi là giả: chia đều phần xác suất giả còn lại.
            w_printed, w_onscreen = 0.5, 0.5

        # Nhân xác suất giả tổng với tỷ trọng của từng loại giả.
        p_printed = p_spoof * w_printed
        p_onscreen = p_spoof * w_onscreen
        total = p_real + p_printed + p_onscreen

        # Chuẩn hóa lần cuối để chống sai số số thực.
        return {
            "real": float(p_real / total),
            "printed": float(p_printed / total),
            "onscreen": float(p_onscreen / total),
        }
