"""Unit test cho src/decision (M1) theo Quy tắc quyết định DD6 & Mục II.7."""

import pytest


def test_decision_empty_scores_returns_no_face():
    """Không có frame nào có mặt phải trả về đúng nhãn 'No_Face'."""
    schemas = pytest.importorskip("src.schemas")
    aggregator = pytest.importorskip("src.decision.aggregator")

    thresholds = {"tau_spoof": 0.60, "tau_deepfake": 0.60, "uncertain_margin": 0.05}
    decision = aggregator.aggregate_decision([], [], thresholds)

    assert isinstance(decision, schemas.FinalDecision)
    assert decision.final_label == "No_Face"


def test_decision_dd6_prioritizes_spoof_over_deepfake():
    """DD6: Nếu điểm spoof > tau_spoof VÀ deepfake > tau_deepfake, phải ưu tiên nhãn Spoof."""
    schemas = pytest.importorskip("src.schemas")
    aggregator = pytest.importorskip("src.decision.aggregator")

    spoof_scores = [
        schemas.SpoofScore(frame_id=0, probs={"Real": 0.10, "Printed": 0.10, "On-screen": 0.80}),
        schemas.SpoofScore(frame_id=1, probs={"Real": 0.12, "Printed": 0.08, "On-screen": 0.80}),
    ]
    df_scores = [
        schemas.DeepfakeScore(frame_id=0, fake_prob=0.92),
        schemas.DeepfakeScore(frame_id=1, fake_prob=0.90),
    ]
    thresholds = {"tau_spoof": 0.60, "tau_deepfake": 0.60, "uncertain_margin": 0.05}
    decision = aggregator.aggregate_decision(spoof_scores, df_scores, thresholds)

    assert decision.final_label == "On-screen"
    assert 0.0 <= decision.confidence <= 1.0


def test_decision_printed_vs_onscreen_level2():
    """Khi spoof vượt tau_spoof, nhãn cuối chọn lớp cao hơn giữa Printed và On-screen."""
    schemas = pytest.importorskip("src.schemas")
    aggregator = pytest.importorskip("src.decision.aggregator")

    spoof_scores = [
        schemas.SpoofScore(frame_id=0, probs={"Real": 0.15, "Printed": 0.70, "On-screen": 0.15}),
        schemas.SpoofScore(frame_id=1, probs={"Real": 0.20, "Printed": 0.65, "On-screen": 0.15}),
    ]
    df_scores = [
        schemas.DeepfakeScore(frame_id=0, fake_prob=0.10),
        schemas.DeepfakeScore(frame_id=1, fake_prob=0.12),
    ]
    thresholds = {"tau_spoof": 0.60, "tau_deepfake": 0.60, "uncertain_margin": 0.05}
    decision = aggregator.aggregate_decision(spoof_scores, df_scores, thresholds)

    assert decision.final_label == "Printed"


def test_decision_deepfake_when_spoof_below_threshold():
    """Khi spoof <= tau_spoof và fake_prob > tau_deepfake, trả về 'Deepfake'."""
    schemas = pytest.importorskip("src.schemas")
    aggregator = pytest.importorskip("src.decision.aggregator")

    spoof_scores = [
        schemas.SpoofScore(frame_id=0, probs={"Real": 0.85, "Printed": 0.08, "On-screen": 0.07}),
        schemas.SpoofScore(frame_id=1, probs={"Real": 0.80, "Printed": 0.10, "On-screen": 0.10}),
    ]
    df_scores = [
        schemas.DeepfakeScore(frame_id=0, fake_prob=0.82),
        schemas.DeepfakeScore(frame_id=1, fake_prob=0.78),
    ]
    thresholds = {"tau_spoof": 0.60, "tau_deepfake": 0.60, "uncertain_margin": 0.05}
    decision = aggregator.aggregate_decision(spoof_scores, df_scores, thresholds)

    assert decision.final_label == "Deepfake"


def test_decision_stage1_without_deepfake_returns_real():
    """GĐ 1 (df_scores=None): nếu không vượt ngưỡng spoof thì trả về 'Real'."""
    schemas = pytest.importorskip("src.schemas")
    aggregator = pytest.importorskip("src.decision.aggregator")

    spoof_scores = [
        schemas.SpoofScore(frame_id=0, probs={"Real": 0.88, "Printed": 0.06, "On-screen": 0.06}),
        schemas.SpoofScore(frame_id=1, probs={"Real": 0.90, "Printed": 0.05, "On-screen": 0.05}),
    ]
    thresholds = {"tau_spoof": 0.60, "tau_deepfake": 0.60, "uncertain_margin": 0.05}
    decision = aggregator.aggregate_decision(spoof_scores, None, thresholds)

    assert decision.final_label == "Real"
    assert decision.final_label in {"Real", "Printed", "On-screen", "Deepfake", "No_Face"}


def test_decision_near_threshold_case():
    """Ca sát ngưỡng: đảm bảo không crash và trả về 1 trong 5 nhãn chuẩn hóa."""
    schemas = pytest.importorskip("src.schemas")
    aggregator = pytest.importorskip("src.decision.aggregator")

    spoof_scores = [
        schemas.SpoofScore(frame_id=0, probs={"Real": 0.39, "Printed": 0.31, "On-screen": 0.30}),
    ]
    df_scores = [schemas.DeepfakeScore(frame_id=0, fake_prob=0.59)]
    thresholds = {"tau_spoof": 0.60, "tau_deepfake": 0.60, "uncertain_margin": 0.05}
    decision = aggregator.aggregate_decision(spoof_scores, df_scores, thresholds)

    assert decision.final_label in {"Real", "Printed", "On-screen", "Deepfake", "No_Face"}
    assert 0.0 <= decision.confidence <= 1.0
