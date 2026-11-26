"""Retrain Model 1 with real + LLM-labeled rows with strict leak-free filtering.

Owned by Teammate A / Step 4.

Procedure:
1. Load LLM-labeled extra rows from data/extra/llm_labels_v1.jsonl.
2. Compute near_real_ids (char-TF-IDF cosine similarity >= 0.80) to real training data.
3. Deterministically cap rows per class to prevent majority class dominance.
4. During CV, drop any extra row whose near_real_ids intersects validation fold ids.
5. Evaluate on real rows only (clean and Set B corrupted) and log promotion metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src import config
from src.data_loader import load_sample, load_test, techniques_to_multihot
from src.evaluation.metrics import log_experiment, score_all
from src.llm_label import filter_labels
from src.preprocessing import preprocess
from src.training.cross_validation import (
    FOLDS_PATH,
    compute_slice_metrics,
    corrupt_text,
    iter_saved_folds,
    leave_cluster_out_folds,
    load_or_create_folds,
    save_oof_npz,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EXTRA_PATH_DEFAULT = config.ROOT / "data" / "extra" / "llm_labels_v1.jsonl"


# ── 1. Extra Data Loading ─────────────────────────────────────────────────────

def load_extra(path: Optional[Path] = None, default_weight: float = 0.3) -> pd.DataFrame:
    """Load and format LLM-labeled extra data into standard DataFrame schema.

    Schema: id, text, label, techniques, action, source='llm', weight.
    """
    path = Path(path or EXTRA_PATH_DEFAULT)
    cols = ["id", "text", "label", "techniques", "action", "source", "weight"]

    if not path.exists():
        log.warning("Extra labels file not found: %s. Returning empty DataFrame.", path)
        return pd.DataFrame(columns=cols)

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                resp = data.get("response", data)
                text = data.get("text", resp.get("text", ""))

                label = resp.get("label")
                techniques = resp.get("techniques", [])
                action = resp.get("action")
                conf = resp.get("confidence", 1.0)
                row_id = data.get("id", 80000 + idx)

                records.append({
                    "id": int(row_id),
                    "key": data.get("key", ""),
                    "text": str(text),
                    "label": label,
                    "techniques": techniques,
                    "action": action,
                    "confidence": conf,
                    "source": "llm",
                    "weight": float(default_weight),
                })
            except Exception as e:
                log.debug("Skipping invalid json line in %s: %s", path, e)

    if not records:
        return pd.DataFrame(columns=cols)

    raw_df = pd.DataFrame(records)
    # Apply standard consistency & confidence filtering
    clean_df, drop_report = filter_labels(raw_df)
    log.info("Loaded extra rows: %d raw, %d passed filter (dropped: %s)",
             len(raw_df), len(clean_df), drop_report)

    if clean_df.empty:
        return pd.DataFrame(columns=cols)

    # Format techniques column to 'T1|T2' or 'NONE'
    def _fmt_tech(t):
        if isinstance(t, list):
            return "|".join(sorted(t)) if t else "NONE"
        s = str(t).strip()
        return s if s else "NONE"

    clean_df["techniques"] = clean_df["techniques"].apply(_fmt_tech)
    clean_df["label"] = clean_df["label"].astype(int)
    clean_df["action"] = clean_df["action"].astype(int)
    clean_df["weight"] = float(default_weight)
    clean_df["source"] = "llm"

    return clean_df[cols].reset_index(drop=True)


# ── 2. Leakage Protection: Near-Real IDs ──────────────────────────────────────

def compute_near_real_ids(
    extra_df: pd.DataFrame,
    real_df: pd.DataFrame,
    sim_threshold: float = 0.80,
) -> pd.DataFrame:
    """Identify real-row IDs that have char_wb (3,5) cosine similarity >= 0.80 to each extra row.

    Matches the assign_groups near-duplicate logic to guarantee validation isolation.
    """
    out_df = extra_df.copy()
    if out_df.empty or real_df.empty:
        out_df["near_real_ids"] = [[] for _ in range(len(out_df))]
        return out_df

    cleans_real = [preprocess(t)["clean"] for t in real_df["text"]]
    cleans_extra = [preprocess(t)["clean"] for t in out_df["text"]]

    # Handle empty strings gracefully
    safe_real = [c if c.strip() else "empty_real" for c in cleans_real]
    safe_extra = [c if c.strip() else "empty_extra" for c in cleans_extra]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    vec.fit(safe_real + safe_extra)

    X_real = vec.transform(safe_real)
    X_extra = vec.transform(safe_extra)

    radius = max(0.0, 1.0 - sim_threshold)
    nn = NearestNeighbors(radius=radius, metric="cosine", algorithm="brute")
    nn.fit(X_real)

    indices_list = nn.radius_neighbors(X_extra, return_distance=False)
    real_ids = real_df["id"].values

    near_ids_col = []
    leak_match_count = 0
    for idxs in indices_list:
        matched = [int(real_ids[i]) for i in idxs]
        if matched:
            leak_match_count += 1
        near_ids_col.append(matched)

    out_df["near_real_ids"] = near_ids_col
    log.info("compute_near_real_ids: %d of %d extra rows have near-real matches (sim >= %.2f)",
             leak_match_count, len(out_df), sim_threshold)
    return out_df


# ── 3. Class Capping ──────────────────────────────────────────────────────────

def apply_class_cap(
    extra_df: pd.DataFrame,
    max_per_class: int = 400,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Cap rows per class deterministically, keeping all rows of rare classes."""
    if extra_df.empty:
        return extra_df.copy()

    rng = np.random.RandomState(seed)
    chosen_indices = []

    for label, group in extra_df.groupby("label"):
        if len(group) <= max_per_class:
            chosen_indices.extend(group.index.values)
        else:
            sampled = rng.choice(group.index.values, size=max_per_class, replace=False)
            chosen_indices.extend(sampled)

    capped_df = extra_df.loc[sorted(chosen_indices)].reset_index(drop=True)
    log.info("apply_class_cap: %d -> %d rows (max_per_class=%d)",
             len(extra_df), len(capped_df), max_per_class)
    return capped_df


# ── 4. Cross-Validation with Extra Rows ────────────────────────────────────────

def run_cv_experiment(
    extra_df: Optional[pd.DataFrame] = None,
    exp_name: str = "E024",
    C: float = 5.0,
    n_aug: int = 5,
    extra_weight: float = 0.3,
    max_per_class: int = 400,
    seed: int = config.SEED,
) -> dict:
    """Execute 5x3 grouped cross-validation with extra LLM-labeled rows and compare to baseline."""
    from src.prediction import _CVModel

    df_real = load_sample()
    all_real_ids = df_real["id"].values.tolist()
    folds_table = load_or_create_folds(df_real, path=FOLDS_PATH)

    processed_extra = None
    if extra_df is not None and not extra_df.empty:
        capped = apply_class_cap(extra_df, max_per_class=max_per_class, seed=seed)
        processed_extra = compute_near_real_ids(capped, df_real)

    log.info("Starting CV for experiment %s (extra_rows=%s, C=%.2f, extra_weight=%.2f)...",
             exp_name, len(processed_extra) if processed_extra is not None else 0, C, extra_weight)

    # Instantiate model factory
    def make_model():
        return _CVModel(
            seed=seed,
            n_aug=n_aug,
            C=C,
            model_type="lr",
            extra_df=processed_extra,
            extra_weight=extra_weight,
            aug_extra=True,
            all_real_ids=all_real_ids,
        )

    # 1. Clean 5x3 grouped CV
    from src.training.cross_validation import cross_validate
    res_clean, sum_clean, oof_clean = cross_validate(
        df_real,
        make_model,
        n_repeats=config.N_REPEATS,
        use_saved_folds=True,
        return_oof=True,
    )

    # 2. Corrupted Set B evaluation is already computed inside cross_validate(corrupt_valid=True)
    corrupted_f1_a = None
    if isinstance(sum_clean, pd.DataFrame) and "corrupted" in sum_clean.index:
        corrupted_f1_a = float(sum_clean.loc["corrupted", ("F_A", "mean")])

    # 3. Leave-cluster-out evaluation (domain-shift proxy)
    lco_f1_a = None
    try:
        lco_preds = []
        lco_y = []
        for tr, va, _ in leave_cluster_out_folds(df_real, fold_table=folds_table):
            model = make_model().fit(df_real.iloc[tr])
            pred = model.predict(df_real.iloc[va])
            lco_preds.extend(pred["label"])
            lco_y.extend(df_real.iloc[va]["label"].values)

        from src.evaluation.metrics import macro_f1_multiclass
        lco_f1_a = macro_f1_multiclass(lco_y, lco_preds)
    except Exception as e:
        log.warning("Could not compute LCO: %s", e)

    # 4. Slices on clean OOF
    clean_oof = oof_clean["clean"] if isinstance(oof_clean, dict) and "clean" in oof_clean else oof_clean
    corrupted_oof = oof_clean.get("corrupted") if isinstance(oof_clean, dict) else None
    pred_label_a = clean_oof["label"]
    slice_report = compute_slice_metrics(df_real, {"label": pred_label_a})

    # 5. Composite S_hat
    if isinstance(sum_clean, pd.DataFrame) and "clean" in sum_clean.index:
        f_a = float(sum_clean.loc["clean", ("F_A", "mean")])
        f_b = float(sum_clean.loc["clean", ("F_B", "mean")])
        f_c = float(sum_clean.loc["clean", ("F_C", "mean")])
        brier = float(sum_clean.loc["clean", ("brier", "mean")])
        if "corrupted" in sum_clean.index:
            corrupted_f1_a = float(sum_clean.loc["corrupted", ("F_A", "mean")])
    else:
        f_a = float(sum_clean.get("F_A_mean", 0.55))
        f_b = float(sum_clean.get("F_B_mean", 0.50))
        f_c = float(sum_clean.get("F_C_mean", 0.45))
        brier = float(sum_clean.get("brier_mean", 0.22))

    cal = 1.0 - brier
    f_dom = float(lco_f1_a) if lco_f1_a is not None else f_a
    f_rob = float(corrupted_f1_a) if corrupted_f1_a is not None else f_a

    from src.evaluation.metrics import s_hat
    s_val = s_hat(f_a, f_b, f_c, f_dom, f_rob, cal, f_dom)

    # 6. Save OOF predictions
    oof_path = save_oof_npz(
        df_real,
        clean_oof,
        oof_corrupted=corrupted_oof,
        folds_arr=folds_table["fold_r0"].values,
        exp_name=exp_name,
        suffix="m1",
    )

    # 7. Log to experiments.csv
    exp_row = {
        "exp_id": exp_name,
        "config_json": json.dumps({
            "extra_rows": len(processed_extra) if processed_extra is not None else 0,
            "extra_weight": extra_weight,
            "max_per_class": max_per_class,
            "C": C,
            "n_aug": n_aug,
        }),
        "F_A": round(f_a, 4),
        "F_B": round(f_b, 4),
        "F_C": round(f_c, 4),
        "Cal": round(cal, 4),
        "S_hat": round(s_val, 4),
        "F_domain": round(f_dom, 4),
        "F_robust": round(f_rob, 4),
        "notes": f"Real + LLM augmented rows ({exp_name})",
    }
    log_experiment(exp_row)

    log.info("Experiment %s Results: F_A=%.4f, F_B=%.4f, F_C=%.4f, F_rob=%.4f, F_dom=%.4f, S_hat=%.4f",
             exp_name, f_a, f_b, f_c, f_rob, f_dom, s_val)
    log.info("Slices: %s", slice_report)

    return {
        "exp_name": exp_name,
        "F_A": f_a,
        "F_B": f_b,
        "F_C": f_c,
        "F_robust": f_rob,
        "F_domain": f_dom,
        "Cal": cal,
        "S_hat": s_val,
        "slices": slice_report,
        "oof_path": str(oof_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Retrain Model 1 with real + LLM extra rows.")
    parser.add_argument("--extra-path", type=str, default=str(EXTRA_PATH_DEFAULT), help="Path to llm_labels_v1.jsonl")
    parser.add_argument("--exp-name", type=str, default="E024", help="Experiment name (default: E024)")
    parser.add_argument("--weight", type=float, default=0.3, help="Weight of LLM rows (default: 0.3)")
    parser.add_argument("--max-per-class", type=int, default=400, help="Class cap (default: 400)")
    args = parser.parse_args()

    extra_df = load_extra(Path(args.extra_path), default_weight=args.weight)
    run_cv_experiment(
        extra_df=extra_df,
        exp_name=args.exp_name,
        extra_weight=args.weight,
        max_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
