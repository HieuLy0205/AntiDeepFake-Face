import cv2
import numpy as np


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + np.exp(-x))


def predict_level2(crop_rgb: np.ndarray) -> tuple[float, float]:
    """
    Phân loại mức 2: Printed (ảnh in trên giấy) vs On-screen (ảnh màn hình).
    Trả về (p_printed, p_onscreen) với tổng bằng 1.0.
    """
    if crop_rgb is None or not isinstance(crop_rgb, np.ndarray) or crop_rgb.size == 0:
        return 0.5, 0.5

    if crop_rgb.ndim == 2:
        rgb = np.repeat(crop_rgb[:, :, None], 3, axis=2)
    elif crop_rgb.ndim == 3 and crop_rgb.shape[2] == 3:
        rgb = crop_rgb
    else:
        return 0.5, 0.5

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    mean_brightness = float(gray.mean())
    color_std = float(np.mean([rgb[:, :, c].std() for c in range(3)]))

    # Dựa trên số đo thực tế: onscreen có std_rgb cao hơn (~48.4 vs ~35.4)
    # và lap_var thấp hơn (~4.1 vs ~14.1)
    score_onscreen = (
        0.12 * (color_std - 41.0)
        - 0.08 * (lap_var - 9.0)
        - 0.01 * (mean_brightness - 142.0)
    )

    p_onscreen = float(np.clip(_sigmoid(score_onscreen), 0.01, 0.99))
    p_printed = float(1.0 - p_onscreen)
    return p_printed, p_onscreen