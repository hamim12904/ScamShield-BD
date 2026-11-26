"""Tests for programmatic corruption augmentation (src/augment.py).

Owned by Teammate B (T-020). Run: pytest -q tests/test_augment.py
"""
import random

import pandas as pd

from src import config
from src.augment import corrupt, expand


def test_seeded_determinism():
    text = "bKash Helpline: Apnar account verify korte OTP din."
    a = corrupt(text, random.Random(42), level="medium", ops="A")
    b = corrupt(text, random.Random(42), level="medium", ops="A")
    assert a == b

    c = corrupt(text, random.Random(42), level="medium", ops="B")
    d = corrupt(text, random.Random(42), level="medium", ops="B")
    assert c == d


def test_delimiter_preserved():
    text = "S: Hello ||| R: hi ||| S: send your PIN now"
    for ops in ("A", "B"):
        for level in ("light", "medium", "heavy"):
            out = corrupt(text, random.Random(123), level=level, ops=ops)
            assert out.count(config.DELIMITER) == text.count(config.DELIMITER)


def test_non_empty_output():
    for txt in ["Verify OTP 123456", "a", "টাকা পাঠান"]:
        for ops in ("A", "B"):
            out = corrupt(txt, random.Random(7), level="heavy", ops=ops)
            assert len(out.strip()) > 0


def test_set_a_and_b_differ():
    text = "Please verify your bKash account number and PIN immediately."
    out_a = corrupt(text, random.Random(42), level="heavy", ops="A")
    out_b = corrupt(text, random.Random(42), level="heavy", ops="B")
    assert out_a != out_b


def test_expand_inherits_group_id_and_metadata():
    df = pd.DataFrame({
        "id": ["101", "102"],
        "text": ["Send OTP now", "Your parcel arrived"],
        "label": [1, 0],
        "techniques": ["T1|T2", "NONE"],
        "action": [1, 0],
        "group_id": [10, 20],
    })
    aug = expand(df, k=3, seed=config.SEED, level="medium", ops="A", weight=0.8)
    assert len(aug) == 6
    assert list(aug["group_id"]) == [10, 10, 10, 20, 20, 20]
    assert (aug["source"] == "aug").all()
    assert (aug["weight"] == 0.8).all()
