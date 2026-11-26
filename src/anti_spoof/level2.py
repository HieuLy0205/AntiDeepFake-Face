"""Phân loại vật lý mức 2 (M4): Phân biệt ảnh in giấy (Printed) và phát lại qua màn hình (On-screen)."""

import cv2
import numpy as np


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + np.exp(-x))


def _fft_grid_peakiness(gray_u8: np.ndarray) -> float:
    """Đo độ nhọn năng lượng tần số trung-cao (dấu hiệu moiré / lưới subpixel màn hình)."""
    gray_f = cv2.resize(gray_u8, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gray_f = gray_f - float(np.mean(gray_f))
    win_1d = np.hanning(128).astype(np.float32)
    window = np.outer(win_1d, win_1d)
    spectrum = np.fft.fftshift(np.fft.fft2(gray_f * window))
    mag = np.log1p(np.abs(spectrum))

    yy, xx = np.ogrid[:128, :128]
    dist = np.sqrt((yy - 63.5) ** 2 + (xx - 63.5) ** 2)
    annulus = mag[(dist >= 12.0) & (dist <= 52.0)]
    if annulus.size == 0:
        return 1.0
    p50 = float(np.percentile(annulus, 50))
    p95 = float(np.percentile(annulus, 95))
    return p95 / (p50 + 0.35)


def predict_level2(crop_rgb: np.ndarray) -> tuple[float, float]:
    """
    Phân loại mức 2: Printed (ảnh in trên giấy) vs On-screen (ảnh màn hình).
    Kết hợp 4 đặc trưng vật lý: phổ FFT moiré/lưới điểm ảnh, phản xạ chói specular,
    dải động tương phản (black-floor lift của giấy in), và độ rực màu đèn nền HSV.
    Trả về (p_printed, p_onscreen) trong [0.01, 0.99] với tổng bằng 1.0.
    """
    if crop_rgb is None or not isinstance(crop_rgb, np.ndarray) or crop_rgb.size == 0:
        return 0.5, 0.5

    if crop_rgb.ndim == 2:
        rgb = np.repeat(crop_rgb[:, :, None], 3, axis=2)
    elif crop_rgb.ndim == 3 and crop_rgb.shape[2] == 3:
        rgb = crop_rgb
    else:
        return 0.5, 0.5

    rgb_128 = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb_128, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_128, cv2.COLOR_RGB2HSV)

    # 1. Phổ FFT moiré / lưới pixel màn hình
    fft_peakiness = _fft_grid_peakiness(gray)

    # 2. Điểm chói phản xạ kính màn hình (sáng rất cao + độ bão hòa thấp) & tương phản đỉnh
    s_chan = hsv[:, :, 1].astype(np.float32)
    v_chan = hsv[:, :, 2].astype(np.float32)
    glare_ratio = float(np.mean((v_chan >= 240.0) & (s_chan <= 45.0)))
    highlight_contrast = float(np.percentile(v_chan, 99) - np.percentile(v_chan, 50)) / 255.0

    # 3. Dải động độ sáng & thiên sắc xanh đèn nền (ảnh in thường bị nâng nền tối / hẹp dải động)
    dynamic_span = float(np.percentile(v_chan, 96) - np.percentile(v_chan, 4)) / 255.0
    rgb_f = rgb_128.astype(np.float32)
    blue_excess = float(np.mean(rgb_f[:, :, 2]) - 0.5 * (np.mean(rgb_f[:, :, 0]) + np.mean(rgb_f[:, :, 1]))) / 255.0

    # 4. Độ bão hòa màu đèn nền phát xạ (HSV S 85th percentile)
    sat_p85 = float(np.percentile(s_chan, 85)) / 255.0

    score_onscreen = (
        1.45 * (fft_peakiness - 1.60)
        + 6.50 * (glare_ratio - 0.015)
        + 1.60 * (highlight_contrast - 0.30)
        + 1.35 * (dynamic_span - 0.58)
        + 1.20 * (sat_p85 - 0.42)
        + 1.50 * blue_excess
    )

    p_onscreen = float(np.clip(_sigmoid(score_onscreen), 0.01, 0.99))
    p_printed = float(1.0 - p_onscreen)
    return p_printed, p_onscreen
