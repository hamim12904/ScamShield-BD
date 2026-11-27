"""Confidence-gated test set pseudo-labeling pipeline (T-032).

Extracts high-confidence, class-balanced pseudo-labeled rows from test.csv
using Model 1 (+ Model 2 agreement when available), applies rule consistency,
filters noise-only records, and formats output for leak-free cross-validation.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src import config
from src.data_loader import load_sample, load_test, techniques_to_multihot
from src.features.tfidf_features import build_text_dict
from src.postprocess import apply_rules_batch
from src.prediction import fit_all
from src.preprocessing import preprocess
from src.training.train_llm_aug import (
    apply_class_cap,
    compute_near_real_ids,
    load_extra,
    run_cv_experiment,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PSEUDO_PATH_DEFAULT = config.ROOT / "data" / "extra" / "pseudo_labels_v1.jsonl"


def mine_pseudo_labels(
    df_test: pd.DataFrame,
    fitted_model,
    conf_threshold: float = 0.85,
    max_per_class: int = 100,
    m2_test: Optional[dict] = None,
    seed: int = config.SEED,
    weight: float = 0.20,
) -> pd.DataFrame:
    """Mine high-confidence pseudo-labels from test data.

    Parameters:
    -----------
    df_test: DataFrame with 'id' and 'text' columns
    fitted_model: Trained FittedModel from fit_all()
    conf_threshold: Minimum Task A probability required for inclusion (default: 0.85)
    max_per_class: Maximum number of rows retained per class (balanced sampling)
    m2_test: Optional Model 2 test predictions dict with 'A', 'B', 'C'
    seed: Random seed for deterministic selection
    weight: Sample weight assigned to pseudo-labeled rows in downstream training
    """
    if df_test.empty:
        return pd.DataFrame(columns=["id", "text", "label", "techniques", "action", "source", "weight", "confidence"])

    # 1. Pre-filter noise-only, empty, or very short messages
    log.info("Filtering degenerate/noise-only rows from %d candidates...", len(df_test))
    clean_indices = []
    clean_texts = []
    for idx, (row_id, text) in enumerate(zip(df_test["id"], df_test["text"])):
        t_str = str(text).strip() if pd.notna(text) else ""
        if len(t_str) < 12:
            continue
        p = preprocess(t_str)
        if p["noise_only"]:
            continue
        clean_indices.append(idx)
        clean_texts.append(t_str)

    sub_test = df_test.iloc[clean_indices].copy().reset_index(drop=True)
    if sub_test.empty:
        log.warning("No test candidates passed pre-filtering.")
        return pd.DataFrame(columns=["id", "text", "label", "techniques", "action", "source", "weight", "confidence"])

    # 2. Extract Task A/B/C probabilities via fitted Model 1 in bounded-memory chunks
    chunk_size = 50_000
    n_sub = len(sub_test)
    p_a_parts, p_b_parts, p_c_parts = [], [], []
    w_lr = float(getattr(fitted_model, "blend_w", 0.5))
    has_svm = getattr(fitted_model, "head_a_svm", None) is not None

    for start in range(0, n_sub, chunk_size):
        end = min(start + chunk_size, n_sub)
        chunk = sub_test.iloc[start:end]
        d = build_text_dict(chunk)
        d["struct_feats"] = None
        X = fitted_model.block.transform(d)

        p_a_lr = fitted_model.head_a.predict_proba(X)
        if has_svm:
            p_a_svm = fitted_model.head_a_svm.predict_proba(X)
            p_a_parts.append(w_lr * p_a_lr + (1.0 - w_lr) * p_a_svm)
        else:
            p_a_parts.append(p_a_lr)
        p_b_parts.append(fitted_model.head_b.predict_proba(X))
        p_c_parts.append(fitted_model.head_c.predict_proba(X))

    p_a = np.vstack(p_a_parts)
    p_b = np.vstack(p_b_parts)
    p_c = np.vstack(p_c_parts)

    # If Model 2 predictions match indices, require agreement
    if m2_test is not None and "A" in m2_test:
        m2_a_full = np.asarray(m2_test["A"])
        # Only use M2 if it aligns with test indices
        if len(m2_a_full) == len(df_test):
            m2_a_sub = m2_a_full[clean_indices]
            # Agreement: argmax of M1 and M2 must match
            m1_pred = p_a.argmax(axis=1)
            m2_pred = m2_a_sub.argmax(axis=1)
            agreement = (m1_pred == m2_pred)
            # Combine probabilities
            p_a = 0.6 * p_a + 0.4 * m2_a_sub
        else:
            agreement = np.ones(len(sub_test), dtype=bool)
    else:
        agreement = np.ones(len(sub_test), dtype=bool)

    # 3. Confidence filter
    max_prob = p_a.max(axis=1)
    pred_label = p_a.argmax(axis=1)

    qualifies = (max_prob >= conf_threshold) & agreement
    qualified_indices = np.where(qualifies)[0]

    log.info(
        "Candidate filtering: %d / %d qualified (conf >= %.2f, agreement=%s)",
        len(qualified_indices), len(sub_test), conf_threshold, m2_test is not None,
    )

    if len(qualified_indices) == 0:
        return pd.DataFrame(columns=["id", "text", "label", "techniques", "action", "source", "weight", "confidence"])

    # 4. Apply multi-task postprocessing rules
    thr_b = np.array(getattr(fitted_model.head_b, "thresholds", [0.35] * 6), dtype=float)
    bin_b = (p_b[qualified_indices] >= thr_b).astype(int)
    pred_c = p_c[qualified_indices].argmax(axis=1)

    labels_post, tech_post, act_post = apply_rules_batch(
        pred_label[qualified_indices],
        bin_b,
        pred_c,
        p_b[qualified_indices],
        p_c[qualified_indices],
    )

    # Convert binary techniques to comma-separated format
    tech_strings = []
    for row in tech_post:
        active = [f"T{j+1}" for j, v in enumerate(row) if v == 1]
        tech_strings.append(",".join(active) if active else "NONE")

    selected_df = pd.DataFrame({
        "id": sub_test.iloc[qualified_indices]["id"].values,
        "text": sub_test.iloc[qualified_indices]["text"].values,
        "label": labels_post.astype(int),
        "techniques": tech_strings,
        "action": act_post.astype(int),
        "source": "pseudo",
        "weight": float(weight),
        "confidence": max_prob[qualified_indices],
    })

    # 5. Deterministic balanced class capping (prioritizing highest confidence per class)
    capped_parts = []
    for lab in sorted(selected_df["label"].unique()):
        class_df = selected_df[selected_df["label"] == lab].sort_values(by="confidence", ascending=False)
        capped_parts.append(class_df.head(max_per_class))

    final_df = pd.concat(capped_parts, ignore_index=True)
    counts = dict(final_df["label"].value_counts())
    log.info("Mined %d pseudo-labeled rows across classes: %s", len(final_df), counts)
    return final_df


def save_pseudo_labels(df: pd.DataFrame, path: Path = PSEUDO_PATH_DEFAULT) -> None:
    """Save pseudo-labeled rows to JSONL format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "id": int(row["id"]),
                "text": str(row["text"]),
                "response": {
                    "label": int(row["label"]),
                    "techniques": row["techniques"].split(",") if row["techniques"] != "NONE" else [],
                    "action": int(row["action"]),
                    "confidence": float(row.get("confidence", 0.90)),
                },
                "source": "pseudo",
                "weight": float(row.get("weight", 0.20)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("Saved %d pseudo-labeled rows to %s", len(df), path)


def load_pseudo_labels(path: Path = PSEUDO_PATH_DEFAULT, weight: float = 0.20) -> pd.DataFrame:
    """Load pseudo-labeled rows from JSONL file."""
    path = Path(path)
    if not path.exists():
        log.warning("Pseudo-labels file not found: %s", path)
        return pd.DataFrame(columns=["id", "text", "label", "techniques", "action", "source", "weight"])

    return load_extra(path, default_weight=weight)
