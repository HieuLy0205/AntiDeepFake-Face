"""Multi-frame probability aggregation (DD7) and hierarchical decision rule (DD6).

Decision precedence (DD6):
1. Zero valid face frames -> `"No_Face"`
2. Aggregated Spoof probability (`Printed + On-screen`) > `tau_spoof` ->
   `"Printed"` if `Printed >= On-screen` else `"On-screen"`
3. Stage 2 enabled and aggregated `fake_prob` > `tau_deepfake` -> `"Deepfake"`
4. Otherwise -> `"Real"` (with uncertainty note if within `uncertain_margin` of threshold)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.schemas import (
    LABEL_DEEPFAKE,
    LABEL_NO_FACE,
    LABEL_ONSCREEN,
    LABEL_PRINTED,
    LABEL_REAL,
    DeepfakeScore,
    FinalDecision,
    SpoofScore,
)


def _robust_mean(values: Sequence[float]) -> float:
    """Compute mean probability across frames (with mild 10% outlier trim when N >= 10)."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) >= 10:
        arr_sorted = np.sort(arr)
        k = max(1, int(len(arr_sorted) * 0.10))
        arr = arr_sorted[k:-k]
    return float(np.clip(np.mean(arr), 0.0, 1.0))


def aggregate_decision(
    spoof_scores: Sequence[SpoofScore],
    df_scores: Optional[Sequence[DeepfakeScore]],
    thresholds: Optional[Dict[str, float]] = None,
    frame_logs: Optional[List[dict]] = None,
    preview_frame_rgb: Optional[np.ndarray] = None,
    preview_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> FinalDecision:
    """Aggregate per-frame Stage 1 & Stage 2 scores into a `FinalDecision` per DD6 & DD7."""
    thr = thresholds or {}
    tau_spoof = float(thr.get("tau_spoof", thr.get("spoof", 0.60)))
    tau_deepfake = float(thr.get("tau_deepfake", thr.get("deepfake", 0.65)))
    margin = float(thr.get("uncertain_margin", 0.08))

    if not spoof_scores:
        return FinalDecision(
            final_label=LABEL_NO_FACE,
            confidence=0.0,
            aggregated_probs={
                LABEL_REAL: 0.0,
                LABEL_PRINTED: 0.0,
                LABEL_ONSCREEN: 0.0,
                LABEL_DEEPFAKE: 0.0,
            },
            reason="Không phát hiện khuôn mặt hợp lệ trong các khung hình.",
            frame_logs=list(frame_logs or []),
            preview_frame_rgb=preview_frame_rgb,
            preview_bbox=preview_bbox,
        )

    mean_real = _robust_mean([float(s.probs.get(LABEL_REAL, 0.0)) for s in spoof_scores])
    mean_printed = _robust_mean([float(s.probs.get(LABEL_PRINTED, 0.0)) for s in spoof_scores])
    mean_onscreen = _robust_mean([float(s.probs.get(LABEL_ONSCREEN, 0.0)) for s in spoof_scores])

    # Re-normalize the 3 anti-spoof classes so they sum cleanly to 1.0
    s_sum = mean_real + mean_printed + mean_onscreen
    if s_sum > 0:
        mean_real /= s_sum
        mean_printed /= s_sum
        mean_onscreen /= s_sum

    spoof_total = float(np.clip(mean_printed + mean_onscreen, 0.0, 1.0))
    stage2_active = df_scores is not None and len(df_scores) > 0
    mean_fake = _robust_mean([float(d.fake_prob) for d in df_scores]) if stage2_active else 0.0  # type: ignore[union-attr]

    agg_probs: Dict[str, float] = {
        LABEL_REAL: round(mean_real, 4),
        LABEL_PRINTED: round(mean_printed, 4),
        LABEL_ONSCREEN: round(mean_onscreen, 4),
        LABEL_DEEPFAKE: round(mean_fake, 4),
    }

    # Precedence 1: Presentation Attack (Spoof > tau_spoof) overrides Deepfake (DD6)
    if spoof_total > tau_spoof:
        if mean_printed >= mean_onscreen:
            final_label = LABEL_PRINTED
            conf = max(mean_printed, spoof_total)
            reason = (
                f"Phát hiện dấu hiệu ảnh in giấy (Spoof={spoof_total * 100:.1f}% > "
                f"ngưỡng {tau_spoof * 100:.0f}%, Printed={mean_printed * 100:.1f}%)."
            )
        else:
            final_label = LABEL_ONSCREEN
            conf = max(mean_onscreen, spoof_total)
            reason = (
                f"Phát hiện dấu hiệu phát lại qua màn hình (Spoof={spoof_total * 100:.1f}% > "
                f"ngưỡng {tau_spoof * 100:.0f}%, On-screen={mean_onscreen * 100:.1f}%)."
            )
    # Precedence 2: Digital Manipulation (Deepfake > tau_deepfake when Stage 2 active)
    elif stage2_active and mean_fake > tau_deepfake:
        final_label = LABEL_DEEPFAKE
        conf = mean_fake
        reason = (
            f"Vượt qua kiểm tra Anti-Spoof (Spoof={spoof_total * 100:.1f}%), nhưng phát hiện "
            f"khuôn mặt tổng hợp/hoán đổi AI (Deepfake={mean_fake * 100:.1f}% > ngưỡng {tau_deepfake * 100:.0f}%)."
        )
    # Precedence 3: Bona fide Real
    else:
        final_label = LABEL_REAL
        conf = min(mean_real, 1.0 - mean_fake) if stage2_active else mean_real
        if stage2_active:
            reason = (
                f"Không phát hiện dấu hiệu giả mạo (Spoof={spoof_total * 100:.1f}%, "
                f"Deepfake={mean_fake * 100:.1f}% đều dưới ngưỡng)."
            )
        else:
            reason = f"Không phát hiện dấu hiệu giả mạo vật lý (Real={mean_real * 100:.1f}%, GĐ 2 đang tắt)."

    near_spoof = margin > 0 and abs(spoof_total - tau_spoof) <= margin
    near_df = margin > 0 and stage2_active and final_label in (LABEL_REAL, LABEL_DEEPFAKE) and abs(mean_fake - tau_deepfake) <= margin
    if near_spoof or near_df:
        reason += " [Lưu ý: Điểm nằm sát ngưỡng quyết định]."

    return FinalDecision(
        final_label=final_label,
        confidence=round(float(np.clip(conf, 0.0, 1.0)), 4),
        aggregated_probs=agg_probs,
        reason=reason,
        frame_logs=list(frame_logs or []),
        preview_frame_rgb=preview_frame_rgb,
        preview_bbox=preview_bbox,
    )
