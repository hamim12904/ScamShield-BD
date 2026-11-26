"""Tests for prediction pipeline (fit_all, predict_df).

Owned by Teammate B (T-010). Run: pytest -q tests/test_predict_df.py
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import load_sample


def _make_df(texts):
    return pd.DataFrame({
        "id":   [str(i) for i in range(len(texts))],
        "text": texts,
    })


@pytest.fixture(scope="module")
def fitted_model():
    """Fit model on sample.csv once for the whole module (no test.csv required)."""
    from src.prediction import fit_all
    return fit_all(load_sample(), seed=config.SEED)


# ── Correctness tests ─────────────────────────────────────────────────────────

def test_basic_prediction_shape_and_columns(fitted_model):
    from src.prediction import predict_df
    df = _make_df([
        "Your OTP is 123456. Do not share.",
        "You won a prize! Click here.",
    ])
    result = predict_df(df, fitted_model)
    assert len(result) == 2
    assert list(result.columns) == ["id", "label", "techniques", "action", "confidence"]


def test_label_range(fitted_model):
    from src.prediction import predict_df
    result = predict_df(_make_df(["hello world", "bKash OTP: 334455"]), fitted_model)
    assert result["label"].between(0, 5).all()


def test_action_range(fitted_model):
    from src.prediction import predict_df
    result = predict_df(_make_df(["hello world"]), fitted_model)
    assert result["action"].between(0, 5).all()


def test_confidence_clipped(fitted_model):
    from src.prediction import predict_df
    result = predict_df(_make_df(["hello world"]), fitted_model)
    lo, hi = config.CONF_CLIP
    assert result["confidence"].between(lo, hi).all()


def test_empty_string_no_error(fitted_model):
    """predict_df handles empty string without raising."""
    from src.prediction import predict_df
    result = predict_df(_make_df([""]), fitted_model)
    assert len(result) == 1


def test_noise_only_no_error(fitted_model):
    """predict_df handles noise-only text without raising."""
    from src.prediction import predict_df
    result = predict_df(_make_df(["wkrjfejejrn zzqwpo xkcdpl"]), fitted_model)
    assert len(result) == 1


def test_long_conversation_no_error(fitted_model):
    """predict_df handles 12-message conversation."""
    from src.prediction import predict_df
    conv = "|||".join([f"Message {i}: some content" for i in range(12)])
    result = predict_df(_make_df([conv]), fitted_model)
    assert len(result) == 1


def test_techniques_format(fitted_model):
    """Techniques column is 'NONE' or valid sorted T-codes joined by '|'."""
    from src.prediction import predict_df
    df = _make_df([
        "bKash: apnar OTP 334455. Eta kauke bolben na.",
        "Delivery: your parcel has arrived.",
        "Hello, how are you?",
    ])
    result = predict_df(df, fitted_model)
    valid_codes = {"T1", "T2", "T3", "T4", "T5", "T6"}
    for t in result["techniques"]:
        if t != "NONE":
            codes = t.split("|")
            assert all(c in valid_codes for c in codes), f"Invalid code in: {t}"
            assert codes == sorted(codes), f"Not sorted: {t}"


def test_id_passthrough(fitted_model):
    """Output ids match input ids exactly."""
    from src.prediction import predict_df
    df = _make_df(["text one", "text two", "text three"])
    result = predict_df(df, fitted_model)
    assert list(result["id"]) == ["0", "1", "2"]
