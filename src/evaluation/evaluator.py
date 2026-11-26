"""ISO/IEC 30107-3 PAD & Deepfake evaluation metrics (APCER, BPCER, ACER, Confusion Matrix)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.schemas import LABEL_DEEPFAKE, LABEL_NO_FACE, LABEL_ONSCREEN, LABEL_PRINTED, LABEL_REAL, FinalDecision


ALL_EVAL_LABELS = [LABEL_REAL, LABEL_PRINTED, LABEL_ONSCREEN, LABEL_DEEPFAKE, LABEL_NO_FACE]


def compute_metrics(
    predictions: Sequence[FinalDecision | str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    """Compute APCER, BPCER, ACER, Accuracy, and multi-class Confusion Matrix.

    Definitions (ISO/IEC 30107-3):
    - Bona fide class: `"Real"`
    - Attack classes: `"Printed"`, `"On-screen"`, `"Deepfake"`
    - `BPCER`: fraction of `"Real"` samples NOT predicted as `"Real"`.
    - `APCER`: fraction of Attack samples erroneously predicted as `"Real"`.
    - `ACER`:  `0.5 * (APCER + BPCER)`.
    """
    if len(predictions) != len(labels):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(labels)} labels")

    pred_labels: List[str] = [
        p.final_label if isinstance(p, FinalDecision) else str(p) for p in predictions
    ]
    true_labels: List[str] = [str(y) for y in labels]

    n_total = len(true_labels)
    n_correct = sum(1 for p, y in zip(pred_labels, true_labels) if p == y)
    accuracy = (n_correct / n_total) if n_total > 0 else 0.0

    real_indices = [i for i, y in enumerate(true_labels) if y == LABEL_REAL]
    attack_indices = [i for i, y in enumerate(true_labels) if y != LABEL_REAL]

    bpcer = (
        sum(1 for i in real_indices if pred_labels[i] != LABEL_REAL) / len(real_indices)
        if real_indices
        else 0.0
    )
    apcer = (
        sum(1 for i in attack_indices if pred_labels[i] == LABEL_REAL) / len(attack_indices)
        if attack_indices
        else 0.0
    )
    acer = 0.5 * (apcer + bpcer)

    present_classes = [c for c in ALL_EVAL_LABELS if c in set(true_labels) | set(pred_labels)]
    cm: Dict[str, Dict[str, int]] = {
        true_c: {pred_c: 0 for pred_c in present_classes} for true_c in present_classes
    }
    for p, y in zip(pred_labels, true_labels):
        cm.setdefault(y, {})
        cm[y][p] = cm[y].get(p, 0) + 1

    return {
        "n_samples": n_total,
        "accuracy": round(float(accuracy), 4),
        "apcer": round(float(apcer), 4),
        "bpcer": round(float(bpcer), 4),
        "acer": round(float(acer), 4),
        "APCER": round(float(apcer), 4),
        "BPCER": round(float(bpcer), 4),
        "ACER": round(float(acer), 4),
        "confusion_matrix": cm,
    }


evaluate = compute_metrics
