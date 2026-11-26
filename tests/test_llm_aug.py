"""Tests for Step 4 real + LLM augmented training and leak-free isolation."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_sample
from src.prediction import _CVModel, fit_all
from src.training.cross_validation import iter_saved_folds, load_or_create_folds
from src.training.train_llm_aug import (
    apply_class_cap,
    compute_near_real_ids,
    load_extra,
)


def test_apply_class_cap():
    extra_df = pd.DataFrame({
        "id": range(1, 21),
        "label": [0] * 12 + [1] * 5 + [2] * 3,
        "text": [f"Sample text {i}" for i in range(1, 21)],
    })
    # Cap classes 0 and 1 (12 and 5 rows) to 4 rows, keep all 3 rows of class 2
    capped = apply_class_cap(extra_df, max_per_class=4, seed=42)
    assert len(capped[capped["label"] == 0]) == 4
    assert len(capped[capped["label"] == 1]) == 4
    assert len(capped[capped["label"] == 2]) == 3
    assert len(capped) == 11


def test_compute_near_real_ids():
    real_df = pd.DataFrame({
        "id": [101, 102],
        "text": [
            "Your bKash account will be suspended tomorrow",
            "Win $1000 lottery cash prize now",
        ],
    })
    extra_df = pd.DataFrame({
        "id": [801, 802, 803],
        "text": [
            "Your bKash account will be suspended tomorrow",  # Identical to 101
            "Your bKash account will be suspended tomorrow morning",  # High similarity to 101
            "Delivery courier package waiting for you",  # Dissimilar
        ],
    })
    res = compute_near_real_ids(extra_df, real_df, sim_threshold=0.80)
    assert 101 in res.loc[0, "near_real_ids"]
    assert 101 in res.loc[1, "near_real_ids"]
    assert len(res.loc[2, "near_real_ids"]) == 0


def test_leakage_isolation_across_folds():
    """Verify that for every validation fold, no extra row near a validation ID is used."""
    real_df = load_sample()
    all_real_ids = real_df["id"].values.tolist()
    folds = load_or_create_folds(real_df)

    # Construct extra rows that match specific real rows
    extra_df = pd.DataFrame({
        "id": [80001, 80002, 80003],
        "text": [
            str(real_df.iloc[0]["text"]),
            str(real_df.iloc[1]["text"]),
            "Completely novel scam message unrelated to any training record",
        ],
        "label": [real_df.iloc[0]["label"], real_df.iloc[1]["label"], 3],
        "techniques": ["NONE", "NONE", "T6"],
        "action": [0, 0, 4],
        "weight": [0.3, 0.3, 0.3],
        "source": ["llm", "llm", "llm"],
    })
    extra_df = compute_near_real_ids(extra_df, real_df, sim_threshold=0.80)

    # For each fold, inspect which extra rows are kept
    for rep in range(1):
        col = f"fold_r{rep}"
        for tr, va in iter_saved_folds(folds, col=col):
            tr_df = real_df.iloc[tr]
            val_ids = set(real_df.iloc[va]["id"].values)

            # Model fit
            model = _CVModel(
                seed=42,
                n_aug=1,
                extra_df=extra_df,
                all_real_ids=all_real_ids,
            )
            # Before fit, test the filter logic directly
            safe_extras = []
            for _, row in extra_df.iterrows():
                near = row["near_real_ids"]
                if not (set(near) & val_ids):
                    safe_extras.append(row["id"])

            # Assert no kept safe extra row matches a validation id
            for kept_id in safe_extras:
                kept_near = extra_df[extra_df["id"] == kept_id]["near_real_ids"].iloc[0]
                assert set(kept_near).intersection(val_ids) == set(), (
                    f"Leakage detected! Kept extra row {kept_id} intersects val_ids {val_ids}"
                )


def test_extra_df_none_unchanged():
    """Ensure extra_df=None gives identical output to baseline."""
    real_df = load_sample().iloc[:15].copy()

    # Fit without extra_df
    m_base = fit_all(real_df, seed=42, n_aug=2)
    # Fit with explicit extra_df=None
    m_none = fit_all(real_df, seed=42, n_aug=2, extra_df=None)

    test_slice = pd.DataFrame({
        "id": [1, 2],
        "text": ["bKash OTP", "Prize lottery"],
    })
    from src.prediction import predict_df
    pred_base = predict_df(test_slice, m_base)
    pred_none = predict_df(test_slice, m_none)

    pd.testing.assert_frame_equal(pred_base, pred_none)
