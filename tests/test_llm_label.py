"""Tests for src/llm_label.py using mocked client (no network)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.llm_label import (
    build_prompt,
    calibrate_prompt_on_real,
    filter_labels,
    label_rows,
    select_templates,
)


def test_build_prompt():
    prompt = build_prompt("Apnar bKash account suspend kora hoyeche. Call 01700000000")
    assert "system" in prompt
    assert "user" in prompt
    # Check that priority order is explicitly included
    assert "1 -> 2 -> 4 -> 3 -> 5 -> 0" in prompt["system"]
    # Check that hard-legit rules are included
    assert "genuine otp" in prompt["system"].lower()
    # Check that technique codes are listed
    assert "T1" in prompt["system"]
    assert "T6" in prompt["system"]
    # Check user prompt contains text
    assert "bKash" in prompt["user"]


def test_select_templates_deterministic():
    test_df = pd.DataFrame({
        "id": range(1, 11),
        "text": [
            "Your order is out for delivery",
            "Win $500 lottery right now",
            "Win $500 lottery right now",  # duplicate template
            "Apnar Nagad account verify korun",
            "Fake job offer part time $50 per day",
            "OTP is 123456. Do not share.",
            "Special 50% discount on recharge",
            "Urgent: your account will be suspended",
            "Short msg",
            "Another short msg",
        ],
    })
    s1 = select_templates(test_df, n=4, seed=42)
    s2 = select_templates(test_df, n=4, seed=42)
    pd.testing.assert_frame_equal(s1, s2)
    assert len(s1) <= 4
    # Ensure template_key deduplication worked
    assert s1["template_key"].nunique() == len(s1)


def test_label_rows_caching_and_resume(tmp_path: Path):
    cache_file = tmp_path / "llm_cache.jsonl"
    calls = []

    def mock_client(prompt):
        text = prompt["user"]
        calls.append(text)
        return {
            "content": json.dumps({
                "label": 1,
                "techniques": ["T2", "T5"],
                "action": 1,
                "confidence": 0.95,
                "language": "banglish",
            }),
            "prompt_tokens": 100,
            "completion_tokens": 20,
        }

    texts = ["Message one", "Message two"]

    # First run: uncached -> 2 calls
    df1 = label_rows(texts, cache_path=cache_file, client_fn=mock_client)
    assert len(df1) == 2
    assert len(calls) == 2
    assert cache_file.exists()

    # Second run (resume): should read all from cache -> 0 new calls
    calls.clear()
    df2 = label_rows(texts, cache_path=cache_file, client_fn=mock_client)
    assert len(df2) == 2
    assert len(calls) == 0

    # Partial resume: add third text -> only 1 new call
    calls.clear()
    df3 = label_rows(texts + ["Message three"], cache_path=cache_file, client_fn=mock_client)
    assert len(df3) == 3
    assert len(calls) == 1


def test_filter_labels():
    raw_df = pd.DataFrame([
        # Valid scam
        {"key": "k1", "text": "t1", "label": 1, "techniques": ["T2"], "action": 1, "confidence": 0.95},
        # Valid legit
        {"key": "k2", "text": "t2", "label": 0, "techniques": [], "action": 0, "confidence": 0.98},
        # Valid promo
        {"key": "k3", "text": "t3", "label": 5, "techniques": [], "action": 0, "confidence": 0.88},
        # Invalid schema (missing action)
        {"key": "k4", "text": "t4", "label": 2, "techniques": ["T4"], "action": None, "confidence": 0.90},
        # Low confidence (< 0.70)
        {"key": "k5", "text": "t5", "label": 1, "techniques": ["T2"], "action": 1, "confidence": 0.65},
        # Inconsistent rule: label 0 with technique
        {"key": "k6", "text": "t6", "label": 0, "techniques": ["T1"], "action": 0, "confidence": 0.92},
        # Inconsistent rule: label 1 with action 0
        {"key": "k7", "text": "t7", "label": 1, "techniques": ["T2"], "action": 0, "confidence": 0.92},
        # Inconsistent rule: label 3 with empty techniques
        {"key": "k8", "text": "t8", "label": 3, "techniques": [], "action": 4, "confidence": 0.92},
        # Duplicate key
        {"key": "k1", "text": "t1", "label": 1, "techniques": ["T2"], "action": 1, "confidence": 0.95},
    ])

    clean_df, reasons = filter_labels(raw_df)
    assert len(clean_df) == 3  # k1, k2, k3
    assert reasons["invalid_schema"] == 1
    assert reasons["low_confidence"] == 1
    assert reasons["inconsistent_rules"] == 3
    assert reasons["duplicate"] == 1


def test_calibrate_prompt_on_real_passing(tmp_path: Path):
    sample_df = pd.DataFrame({
        "id": [101, 102, 103],
        "text": ["Real OTP test msg", "Fake prize lottery win", "bKash PIN phishing claim"],
        "label": [0, 2, 1],
        "techniques": ["NONE", "T4", "T2|T5"],
        "action": [0, 3, 1],
    })

    # Perfect prediction mock
    def perfect_client(prompt):
        text = prompt["user"]
        if "Real OTP" in text:
            pred = {"label": 0, "techniques": [], "action": 0, "confidence": 0.95, "language": "english"}
        elif "Fake prize" in text:
            pred = {"label": 2, "techniques": ["T4"], "action": 3, "confidence": 0.90, "language": "english"}
        else:
            pred = {"label": 1, "techniques": ["T2", "T5"], "action": 1, "confidence": 0.92, "language": "banglish"}
        return {"content": json.dumps(pred)}

    cache_file = tmp_path / "calib_pass.jsonl"
    res = calibrate_prompt_on_real(
        sample_df=sample_df, client_fn=perfect_client, min_macro_f1=0.80, cache_path=cache_file
    )
    assert res["F_A"] >= 0.80
    assert "confusion_matrix" in res


def test_calibrate_prompt_on_real_failing(tmp_path: Path):
    sample_df = pd.DataFrame({
        "id": [101, 102, 103],
        "text": ["Real OTP fail msg", "Fake prize fail msg", "bKash PIN fail msg"],
        "label": [0, 2, 1],
        "techniques": ["NONE", "T4", "T2|T5"],
        "action": [0, 3, 1],
    })

    # Bad prediction mock that classifies everything as 0
    def bad_client(prompt):
        pred = {"label": 0, "techniques": [], "action": 0, "confidence": 0.95, "language": "english"}
        return {"content": json.dumps(pred)}

    cache_file = tmp_path / "calib_fail.jsonl"
    with pytest.raises(RuntimeError, match="CALIBRATION GATE FAILED"):
        calibrate_prompt_on_real(
            sample_df=sample_df, client_fn=bad_client, min_macro_f1=0.80, cache_path=cache_file
        )
