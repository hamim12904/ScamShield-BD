"""Tests for aggregate-only test-set diagnosis helpers."""
import numpy as np
import pandas as pd
import pytest

from src import config
from tools.diagnose_test import (
    confidence_stats,
    cue_rates,
    histograms,
    language_mix,
    m1_m2_disagreement,
    m2_coverage,
    render_report,
)


def _prep_frame():
    return pd.DataFrame({
        "preserved": [
            "hello",
            "OTP 482913 send 500 tk urgent",
            "congrats prize",
            "hello",
        ],
        "noise_only": [True, False, False, False],
        "has_url": [False, True, False, False],
        "n_messages": [1, 2, 1, 1],
    })


def test_m2_coverage_detects_padded_tail():
    probs = np.array([[0.1, 0.9], [0.7, 0.3], [0.7, 0.3]])
    ood = np.array([0.5, 0.2, 0.2])
    out = m2_coverage(probs, ood)

    assert out["n"] == 3
    assert out["padded"] == 2
    assert out["real"] == 1
    assert out["padded_pct"] == pytest.approx(66.6667, abs=0.01)
    assert out["ood_constant_tail"] is True


def test_histograms_counts_labels_actions_and_techniques():
    out = histograms([0, 0, 1], [0, 0, 2], ["NONE", "T1|T2", "NONE"])

    assert out["label"][0] == 2
    assert out["label"][1] == 1
    assert out["action"][2] == 1
    assert out["techniques"]["NONE"] == 2
    assert out["techniques"]["T1"] == 1
    assert out["techniques"]["T2"] == 1
    assert len(out["label"]) == config.NUM_CLASSES


def test_cue_rates_split_predicted_zero_from_one_to_four():
    out = cue_rates(_prep_frame(), [0, 0, 1, 1])

    p0 = out["pred_0"]
    assert p0["n"] == 2
    assert p0["noise_only"]["count"] == 1
    assert p0["has_url"]["pct"] == pytest.approx(50.0)
    assert p0["otp_like"]["count"] == 1
    assert p0["money"]["count"] == 1
    assert p0["cred"]["count"] == 1
    assert p0["urgency"]["count"] == 1
    assert p0["conversation"]["count"] == 1

    p14 = out["pred_1_4"]
    assert p14["n"] == 2
    assert p14["reward"]["count"] == 1
    assert p14["noise_only"]["count"] == 0


def test_confidence_stats_and_language_mix_are_aggregate_only():
    conf = confidence_stats([0, 0, 1], [0.2, 0.4, 0.9])
    assert conf["0"]["mean"] == pytest.approx(0.3)
    assert conf["all"]["n"] == 3

    mix = language_mix(
        ["hello", "ami apnar korte"],
        ["hello", "আপনার", "ami apnar korte"],
        [0, 0, 1],
    )
    assert mix["sample"]["english"]["count"] == 1
    assert mix["sample"]["banglish"]["count"] == 1
    assert mix["test"]["bangla_script"]["count"] == 1
    assert mix["pred_by_bucket"]["bangla_script"]["label_counts"]["0"] == 1


def test_m1_m2_disagreement_skips_padded_rows():
    m1 = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    m2 = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.1, 0.9, 0.0]])
    padded = np.array([False, False, True])
    out = m1_m2_disagreement(m1, m2, padded)

    assert out["available"] is True
    assert out["n"] == 2
    assert out["disagreements"] == 0
    assert m1_m2_disagreement()["available"] is False


def test_render_report_has_h1_status_and_no_row_identifiers():
    labels = [0, 0, 1, 1]
    diagnostics = {
        "coverage": {
            "n": 3, "padded": 2, "real": 1,
            "padded_pct": 66.67, "real_pct": 33.33,
            "ood_equal_last": 2, "ood_equal_last_pct": 66.67,
            "ood_tail_unique": 1, "ood_constant_tail": True,
        },
        "sample_hist": histograms(labels, [0, 0, 1, 1], ["NONE"] * 4),
        "pred_hist": histograms(labels, [0, 0, 1, 1], ["NONE"] * 4),
        "cue": cue_rates(_prep_frame(), labels),
        "conf": confidence_stats(labels, [0.2, 0.3, 0.8, 0.9]),
        "lang": language_mix(["hello"], ["hello", "আপনার"], [0, 0]),
        "disagree": m1_m2_disagreement(),
    }
    report = render_report(diagnostics)

    assert "M2 covers all test rows: no" in report
    assert "H1 M2 probabilities mostly padded:** supported" in report
    assert "H3 Predicted-0 rows contain scam cues:** cannot tell" in report
    assert "66.67%" in report
    assert "test-id-xyz" not in report
