"""Unit test cho src/evaluation (M2) kiểm chứng APCER, BPCER, ACER trên dữ liệu giả có đáp án."""

import math
import pytest


def test_evaluation_metrics_apcer_bpcer_acer_exact(tmp_path):
    """Kiểm chứng công thức APCER, BPCER, ACER với tập dự đoán biết trước.

    Giả sử 4 mẫu:
    - 2 Real (Bona fide): 1 đoán đúng Real, 1 đoán nhầm Printed -> BPCER = 1/2 = 0.50
    - 2 Attack (Printed, On-screen): 1 đoán nhầm Real, 1 đoán đúng On-screen -> APCER = 1/2 = 0.50
    - ACER = (APCER + BPCER) / 2 = 0.50
    """
    evaluator = pytest.importorskip("src.evaluation.evaluator")
    schemas = pytest.importorskip("src.schemas")

    # Kiểm tra nếu M2 tách hàm compute_metrics(predictions, labels) hoặc evaluate(...)
    compute_fn = getattr(evaluator, "compute_metrics", getattr(evaluator, "evaluate", None))
    if compute_fn is None:
        pytest.skip("src.evaluation.evaluator chưa định nghĩa compute_metrics hoặc evaluate.")

    predictions = [
        schemas.FinalDecision(
            final_label="Real",
            confidence=0.9,
            aggregated_probs={"Real": 0.9, "Printed": 0.05, "On-screen": 0.05, "Deepfake": 0.0},
            reason="ok",
        ),
        schemas.FinalDecision(
            final_label="Printed",
            confidence=0.8,
            aggregated_probs={"Real": 0.1, "Printed": 0.8, "On-screen": 0.1, "Deepfake": 0.0},
            reason="false reject",
        ),
        schemas.FinalDecision(
            final_label="Real",
            confidence=0.7,
            aggregated_probs={"Real": 0.7, "Printed": 0.2, "On-screen": 0.1, "Deepfake": 0.0},
            reason="false accept",
        ),
        schemas.FinalDecision(
            final_label="On-screen",
            confidence=0.85,
            aggregated_probs={"Real": 0.05, "Printed": 0.1, "On-screen": 0.85, "Deepfake": 0.0},
            reason="ok",
        ),
    ]
    labels = ["Real", "Real", "Printed", "On-screen"]

    metrics = compute_fn(predictions, labels)
    assert isinstance(metrics, dict)

    apcer = metrics.get("apcer", metrics.get("APCER"))
    bpcer = metrics.get("bpcer", metrics.get("BPCER"))
    acer = metrics.get("acer", metrics.get("ACER"))

    assert apcer is not None and math.isclose(float(apcer), 0.5, abs_tol=1e-4)
    assert bpcer is not None and math.isclose(float(bpcer), 0.5, abs_tol=1e-4)
    assert acer is not None and math.isclose(float(acer), 0.5, abs_tol=1e-4)
    assert "confusion_matrix" in metrics
