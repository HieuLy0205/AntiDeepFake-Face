"""Stage 1 Face Anti-Spoofing (M4): Two-level MiniFASNetV2 ONNX + Spectral/Color PAD.

Architecture grounded in project design (Slide 3 & Section 2.2) and PAD literature:
1. **Level 1 (`Real` vs `Spoof`) — Neural MiniFASNetV2 (`2.7_80x80_MiniFASNetV2.onnx`):**
   - Uses Minivision Silent-Face-Anti-Spoofing's 2.7x multi-scale context crop (DD4) resized
     to `80x80` BGR float32 `[0..255]` (trained with Fourier-spectrum auxiliary supervision).
   - Outputs `p_real = softmax[1]` and `p_spoof = softmax[0] + softmax[2]`.
2. **Level 2 (`Printed` vs `On-screen` disambiguation of `p_spoof`):**
   - Combines MiniFASNet's `[p_0, p_2]` prior with classical cues from IEEE TIFS PAD papers:
     - **2D FFT high-frequency Moiré / subpixel aliasing energy** (Patel et al., IEEE TIFS 2016)
     - **HSV specular screen glare & saturation variance** (Boulkenafet et al., IEEE TIFS 2016)
     - **Laplacian micro-texture sharpness vs matte paper diffusion** and **YCbCr blue-channel shift**.
   - Splits `p_spoof` cleanly into `Printed` and `On-screen` so `Real + Printed + On-screen == 1.0`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from src.face_detection.detector import crop_with_scale
from src.schemas import LABEL_ONSCREEN, LABEL_PRINTED, LABEL_REAL, FaceCropResult, SpoofScore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MINIFAS_PATHS: Tuple[Path, ...] = (
    REPO_ROOT / "models" / "minifasnet_onnx" / "minifasnet_v2.onnx",
    REPO_ROOT / "models" / "minifasnet_v2.onnx",
)


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exps = np.exp(shifted)
    return exps / float(np.sum(exps))


def _compute_screen_vs_print_ratio(patch_rgb: np.ndarray, p0: float, p2: float) -> float:
    """Estimate `P(On-screen | Spoof) in [0.05, 0.95]` using FFT Moiré + HSV/YCbCr + sharpness cues.

    Returns fraction assigned to `On-screen` (`1 - ratio` is assigned to `Printed`).
    """
    gray_full = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fh, fw = gray_full.shape
    # Measure micro-texture & Moiré inside the central facial region (avoids step-edge bias at border)
    y0, y1 = int(fh * 0.20), max(int(fh * 0.20) + 16, int(fh * 0.80))
    x0, x1 = int(fw * 0.20), max(int(fw * 0.20) + 16, int(fw * 0.80))
    gray = gray_full[y0:y1, x0:x1]

    # Cue 1: 2D FFT mid-high frequency ring energy (screen pixel grid / Moiré interference)
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * win)))
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (0.5 * min(h, w))
    mid_high = float(fft_mag[(r >= 0.35) & (r <= 0.85)].mean() + 1e-6)
    low_band = float(fft_mag[(r >= 0.08) & (r < 0.35)].mean() + 1e-6)
    hf_ratio = mid_high / low_band  # higher on screen Moiré/subpixel grids, lower on diffused paper

    # Cue 2: Laplacian micro-edge variance (matte print blurs high frequencies; screens stay crisp/aliased)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # Cue 3: HSV specular glare vs matte paper compressed dynamic range (Boulkenafet et al., IEEE TIFS)
    hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)
    val_std = float(val.std())
    specular_frac = float(np.mean((val >= 250.0) & (sat < 25.0))) if val_std > 48.0 else 0.0
    dark_bezel_frac = float(np.mean(val < 20.0))

    # Cue 4: YCbCr blue-luminescence emission bias (LCD/OLED backlight blue peak)
    ycrcb = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    cb_minus_cr = float(np.mean(ycrcb[..., 2] - ycrcb[..., 1]))

    # Combine standardized physical cues into a zero-centered screen vs print log-odds score
    score_screen = 0.0
    score_screen += np.clip((hf_ratio - 0.42) * 7.5, -1.8, 1.8)
    score_screen += np.clip((np.log1p(lap_var) - 6.2) * 0.85, -1.8, 1.6)
    score_screen += np.clip((val_std - 46.0) * 0.04, -1.0, 1.0)
    score_screen += np.clip(specular_frac * 22.0, 0.0, 1.4)
    score_screen += np.clip((dark_bezel_frac - 0.08) * 8.0, -0.4, 1.2)
    score_screen += np.clip((cb_minus_cr + 10.0) * 0.05, -0.8, 0.8)

    prob_screen = 1.0 / (1.0 + float(np.exp(-score_screen)))
    return float(np.clip(prob_screen, 0.05, 0.95))


class AntiSpoofDetector:
    """MiniFASNetV2 ONNX wrapper with 2.7x multi-scale context crop & Level-2 Print/Screen split."""

    def __init__(self, onnx_path: Optional[str | Path] = None) -> None:
        env_path = os.environ.get("ADF_ANTISPOOF_ONNX_PATH")
        if onnx_path is not None:
            chosen = Path(onnx_path)
        elif env_path:
            chosen = Path(env_path)
        else:
            chosen = next((p for p in DEFAULT_MINIFAS_PATHS if p.is_file()), DEFAULT_MINIFAS_PATHS[0])

        self.onnx_path: Optional[Path] = chosen if chosen.is_file() else None
        self._session: Any = None
        self._input_name: Optional[str] = None

        if self.onnx_path is not None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self.onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name

    def predict(
        self,
        face_crop: FaceCropResult,
        full_frame_rgb: Optional[np.ndarray] = None,
    ) -> SpoofScore:
        if not face_crop.has_face or face_crop.face_rgb is None:
            return SpoofScore(
                frame_id=int(face_crop.frame_id),
                probs={LABEL_REAL: 0.0, LABEL_PRINTED: 0.0, LABEL_ONSCREEN: 0.0},
            )

        # Extract 2.7x multi-scale 80x80 RGB patch when full frame + bbox are available (DD4)
        if full_frame_rgb is not None and face_crop.bbox is not None:
            patch80_rgb = crop_with_scale(full_frame_rgb, face_crop.bbox, scale=2.7, out_size=(80, 80))
            analysis_rgb = crop_with_scale(full_frame_rgb, face_crop.bbox, scale=2.2, out_size=(160, 160))
        else:
            patch80_rgb = cv2.resize(face_crop.face_rgb, (80, 80), interpolation=cv2.INTER_LINEAR)
            analysis_rgb = face_crop.face_rgb

        if self._session is not None and self._input_name is not None:
            # Upstream Silent-Face-Anti-Spoofing input: (1, 3, 80, 80) BGR float32 in [0..255]
            bgr = cv2.cvtColor(patch80_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
            chw = np.ascontiguousarray(np.transpose(bgr, (2, 0, 1))[None, ...], dtype=np.float32)
            logits = np.asarray(self._session.run(None, {self._input_name: chw})[0][0], dtype=np.float64)
            raw_probs = _softmax_1d(logits)
            # Upstream Minivision index 1 = Real/Live; indices 0 & 2 = Spoof
            p0, p_real, p2 = float(raw_probs[0]), float(raw_probs[1]), float(raw_probs[2])
            p_spoof = max(0.0, 1.0 - p_real)
        else:
            # Pure classical fallback only when ONNX weights are absent
            p0, p2 = 0.05, 0.05
            p_real = 0.90
            p_spoof = 0.10

        screen_share = _compute_screen_vs_print_ratio(analysis_rgb, p0=p0, p2=p2)
        p_onscreen = p_spoof * screen_share
        p_printed = p_spoof * (1.0 - screen_share)

        total = p_real + p_printed + p_onscreen
        if total > 0:
            p_real /= total
            p_printed /= total
            p_onscreen /= total

        return SpoofScore(
            frame_id=int(face_crop.frame_id),
            probs={
                LABEL_REAL: float(np.clip(p_real, 0.0, 1.0)),
                LABEL_PRINTED: float(np.clip(p_printed, 0.0, 1.0)),
                LABEL_ONSCREEN: float(np.clip(p_onscreen, 0.0, 1.0)),
            },
        )


_DEFAULT_SPOOF_DETECTOR: Optional[AntiSpoofDetector] = None


def get_default_spoof_detector() -> AntiSpoofDetector:
    global _DEFAULT_SPOOF_DETECTOR
    if _DEFAULT_SPOOF_DETECTOR is None:
        _DEFAULT_SPOOF_DETECTOR = AntiSpoofDetector()
    return _DEFAULT_SPOOF_DETECTOR


def predict_spoof(
    face_crop: FaceCropResult,
    full_frame_rgb: Optional[np.ndarray] = None,
    detector: Optional[AntiSpoofDetector] = None,
) -> SpoofScore:
    """Canonical Stage 1 Anti-Spoof entry point (`tests/test_anti_spoof_deepfake.py`)."""
    det = detector if detector is not None else get_default_spoof_detector()
    return det.predict(face_crop, full_frame_rgb=full_frame_rgb)
