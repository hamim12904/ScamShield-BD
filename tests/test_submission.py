"""Submission validator tests.

Owned by Teammate B (seed for C's full test_submission.py). pytest -q
Fixtures use synthetic ids, never test rows (competition rule A3).
"""
import pandas as pd
import pytest


def _write_csv(data: dict, path) -> None:
    pd.DataFrame(data).to_csv(path, index=False)


def _validate(path, sample_path=None, test_path=None):
    from tools.validate_submission import validate
    return validate(str(path), sample_path, test_path)


# ── Good submission ───────────────────────────────────────────────────────────

def test_good_submission_passes(tmp_path):
    p = tmp_path / "good.csv"
    _write_csv({
        "id":         ["100", "101"],
        "label":      ["0", "1"],
        "techniques": ["NONE", "T1|T2"],
        "action":     ["0", "1"],
        "confidence": ["0.5", "0.9"],
    }, p)
    errors, _ = _validate(p)
    assert errors == [], errors


# ── One bad pattern each ──────────────────────────────────────────────────────

def test_float_label_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    _write_csv({"id": ["1"], "label": ["2.0"], "techniques": ["NONE"],
                "action": ["0"], "confidence": ["0.5"]}, p)
    errors, _ = _validate(p)
    assert any("label" in e.lower() for e in errors)


def test_empty_techniques_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    _write_csv({"id": ["1"], "label": ["0"], "techniques": [""],
                "action": ["0"], "confidence": ["0.5"]}, p)
    errors, _ = _validate(p)
    assert errors


def test_confidence_above_1_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    _write_csv({"id": ["1"], "label": ["0"], "techniques": ["NONE"],
                "action": ["0"], "confidence": ["1.5"]}, p)
    errors, _ = _validate(p)
    assert any("confidence" in e.lower() for e in errors)


def test_wrong_column_order_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({
        "id": ["1"], "confidence": ["0.5"],
        "label": ["0"], "techniques": ["NONE"], "action": ["0"],
    }).to_csv(p, index=False)
    errors, _ = _validate(p)
    assert any("column" in e.lower() for e in errors)


def test_duplicate_id_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    _write_csv({"id": ["1", "1"], "label": ["0", "0"], "techniques": ["NONE", "NONE"],
                "action": ["0", "0"], "confidence": ["0.5", "0.5"]}, p)
    errors, _ = _validate(p)
    assert any("duplicate" in e.lower() for e in errors)


def test_confidence_nan_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    with open(p, "w") as f:
        f.write("id,label,techniques,action,confidence\n1,0,NONE,0,\n")
    errors, _ = _validate(p)
    assert errors


def test_unsorted_techniques_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    _write_csv({"id": ["1"], "label": ["1"], "techniques": ["T2|T1"],
                "action": ["1"], "confidence": ["0.8"]}, p)
    errors, _ = _validate(p)
    assert errors


def test_invalid_technique_code_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    _write_csv({"id": ["1"], "label": ["1"], "techniques": ["T1|T7"],
                "action": ["1"], "confidence": ["0.8"]}, p)
    errors, _ = _validate(p)
    assert errors


def test_all_none_techniques_ok(tmp_path):
    """NONE is always valid for any label (rules enforced in postprocess, not here)."""
    p = tmp_path / "good.csv"
    _write_csv({"id": ["1"], "label": ["1"], "techniques": ["NONE"],
                "action": ["0"], "confidence": ["0.5"]}, p)
    errors, _ = _validate(p)
    assert errors == [], errors
