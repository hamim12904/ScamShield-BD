"""Systematic tuning of Model 1 (TF-IDF + linear models) on saved 5x3 grouped CV.

Owned by Teammate B (T-021).
Run: python -m src.training.tune_m1
"""
import json
import logging

import numpy as np
import pandas as pd

from src import config
from src.augment import expand
from src.data_loader import load_sample, techniques_to_multihot
from src.evaluation.metrics import log_experiment, s_hat
from src.features.tfidf_features import TfidfBlock, build_text_dict
from src.models.task_a import TaskAHead
from src.models.task_b import TaskBHead
from src.models.task_c import TaskCHead
from src.postprocess import apply_rules_batch, tune_task_b_thresholds
from src.training.cross_validation import cross_validate, leave_cluster_out_cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class TunableM1:
    """Configurable Model 1 wrapper for CV and augmentation experiments."""

    def __init__(self,
                 C: float = 5.0,
                 model_type: str = "lr",
                 use_word: bool = True,
                 use_char: bool = True,
                 use_char_raw: bool = False,
                 use_msg_blocks: bool = True,
                 use_struct: bool = True,
                 n_aug: int = 0,
                 aug_levels: tuple = ("light", "medium", "heavy"),
                 aug_weights: tuple = (1.0, 0.8, 0.5),
                 seed: int = config.SEED):
        self.C = C
        self.model_type = model_type
        self.use_word = use_word
        self.use_char = use_char
        self.use_char_raw = use_char_raw
        self.use_msg_blocks = use_msg_blocks
        self.use_struct = use_struct
        self.n_aug = n_aug
        self.aug_levels = aug_levels
        self.aug_weights = aug_weights
        self.seed = seed

    def _prepare_dict(self, df: pd.DataFrame) -> dict:
        d = build_text_dict(df)
        if not self.use_struct:
            d["struct_feats"] = None
        return d

    def fit(self, train: pd.DataFrame) -> "TunableM1":
        parts = [train.assign(weight=1.0)]
        if self.n_aug > 0:
            for i in range(self.n_aug):
                lvl = self.aug_levels[i % len(self.aug_levels)]
                w = self.aug_weights[i % len(self.aug_weights)]
                aug_df = expand(train, k=1, seed=self.seed + 100 * (i + 1),
                                level=lvl, ops="A", weight=w)
                parts.append(aug_df)
        full_train = pd.concat(parts, ignore_index=True)
        weights = full_train["weight"].values.astype(float)

        train_d = self._prepare_dict(full_train)
        self.block = TfidfBlock(
            max_word_features=200_000 if self.use_word else 1,
            max_char_features=200_000 if self.use_char else 1,
            use_char_raw=self.use_char_raw,
            use_msg_blocks=self.use_msg_blocks,
        )
        X_tr = self.block.fit_transform(train_d)

        y_a = full_train["label"].values
        Y_b = techniques_to_multihot(full_train["techniques"])
        y_c = full_train["action"].values

        self.head_a = TaskAHead(C=self.C, model_type=self.model_type, seed=self.seed).fit(
            X_tr, y_a, sample_weight=weights
        )
        self.head_b = TaskBHead(C=self.C, seed=self.seed).fit(
            X_tr, Y_b, sample_weight=weights
        )
        self.head_c = TaskCHead(C=self.C, seed=self.seed).fit(
            X_tr, y_c, sample_weight=weights
        )
        return self

    def predict(self, valid: pd.DataFrame) -> dict:
        valid_d = self._prepare_dict(valid)
        X_va = self.block.transform(valid_d)
        proba_a = self.head_a.predict_proba(X_va)
        proba_b = self.head_b.predict_proba(X_va)
        proba_c = self.head_c.predict_proba(X_va)
        return {
            "proba":   proba_a,
            "proba_b": proba_b,
            "proba_c": proba_c,
            "label":   proba_a.argmax(1),
            "tech":    self.head_b.predict(X_va),
            "action":  proba_c.argmax(1),
        }


def run_exp(df, exp_id: str, cfg: dict, notes: str = "") -> dict:
    """Run 5x3 grouped CV + leave-cluster-out proxy, log to experiments.csv, return metrics."""
    def make():
        return TunableM1(**cfg)

    res, summary, oof_bundle = cross_validate(
        df, make,
        n_splits=config.N_SPLITS,
        n_repeats=config.N_REPEATS,
        corrupt_valid=True,
        use_saved_folds=True,
        exp_name=exp_id,
        model_tag="m1",
        return_oof=True,
    )
    lco_scores, _ = leave_cluster_out_cv(df, make, saved_folds=oof_bundle["folds"])
    f_domain = float(lco_scores["F_A"])

    clean = summary.loc["clean"]
    corr = summary.loc["corrupted"]
    f_a = float(clean[("F_A", "mean")])
    f_b = float(clean[("F_B", "mean")])
    f_c = float(clean[("F_C", "mean")])
    cal = float(1.0 - clean[("brier", "mean")])
    f_rob = float(corr[("F_A", "mean")])
    sh = s_hat(f_a, f_b, f_c, f_domain, f_rob, cal, f_domain)

    slices_df = oof_bundle["slices"]
    clean_slices = slices_df[slices_df["view"] == "clean"].mean(numeric_only=True)

    row = {
        "exp_id": exp_id,
        "config_json": json.dumps(cfg, sort_keys=True),
        "F_A": round(f_a, 4),
        "F_B": round(f_b, 4),
        "F_C": round(f_c, 4),
        "Cal": round(cal, 4),
        "S_hat": round(sh, 4),
        "F_domain": round(f_domain, 4),
        "F_robust": round(f_rob, 4),
        "notes": notes,
    }
    for col in clean_slices.index:
        if col.startswith("F_A_"):
            row[col] = round(float(clean_slices[col]), 4)

    log_experiment(row)
    print(
        f"[{exp_id}] S_hat={sh:.3f} | F_A={f_a:.3f} | F_rob={f_rob:.3f} | "
        f"F_dom={f_domain:.3f} | F_B={f_b:.3f} | F_C={f_c:.3f} | {notes}"
    )
    return row


def main():
    df = load_sample()
    experiments = [
        # 1. Feature blocks (baseline C=5)
        ("E002", dict(C=5.0, use_msg_blocks=False, use_struct=False), "word+char only"),
        ("E003", dict(C=5.0, use_msg_blocks=True,  use_struct=True),  "word+char+msg+struct (baseline)"),
        ("E004", dict(C=5.0, use_char_raw=True, use_msg_blocks=True, use_struct=True), "+char_raw"),
        # 2. C grid on baseline
        ("E005", dict(C=1.0,  use_msg_blocks=True, use_struct=True), "C=1.0"),
        ("E006", dict(C=2.0,  use_msg_blocks=True, use_struct=True), "C=2.0"),
        ("E007", dict(C=10.0, use_msg_blocks=True, use_struct=True), "C=10.0"),
        ("E008", dict(C=20.0, use_msg_blocks=True, use_struct=True), "C=20.0"),
        # 3. Augmentation Set A (n_aug variants)
        ("E009", dict(C=5.0,  n_aug=2), "C=5, n_aug=2"),
        ("E010", dict(C=5.0,  n_aug=3), "C=5, n_aug=3"),
        ("E011", dict(C=10.0, n_aug=3), "C=10, n_aug=3"),
        ("E012", dict(C=5.0,  n_aug=5), "C=5, n_aug=5"),
        # 4. LinearSVC comparison
        ("E013", dict(C=1.0, model_type="svm", n_aug=3), "LinearSVC C=1, n_aug=3"),
        ("E014", dict(C=0.5, model_type="svm", n_aug=3), "LinearSVC C=0.5, n_aug=3"),
    ]

    results = []
    for exp_id, cfg, notes in experiments:
        results.append(run_exp(df, exp_id, cfg, notes))

    res_df = pd.DataFrame(results).sort_values("S_hat", ascending=False)
    print("\n=== TOP EXPERIMENTS BY S_hat ===")
    print(res_df[["exp_id", "S_hat", "F_A", "F_robust", "F_domain", "F_B", "F_C", "notes"]].to_string(index=False))

    # Tune Task B thresholds & check postprocess gain on best OOF
    best_id = res_df.iloc[0]["exp_id"]
    oof_path = config.OOF_DIR / f"{best_id}_m1.npz"
    oof = np.load(oof_path, allow_pickle=True)
    Y_true = techniques_to_multihot(df["techniques"])
    thrs = tune_task_b_thresholds(oof["B"], Y_true, oof["folds"])
    tech_tuned = (oof["B"] >= np.array(thrs)).astype(int)
    lbl, tech_pp, act_pp = apply_rules_batch(
        oof["A"].argmax(1), tech_tuned, oof["C"].argmax(1),
        oof["B"], oof["C"],
    )
    from src.evaluation.metrics import macro_f1_multiclass, macro_f1_techniques
    print(f"\n=== POSTPROCESS CHECK ON {best_id} OOF ===")
    print("Tuned thresholds:", thrs)
    print("F_B raw (0.5):", round(macro_f1_techniques(Y_true, (oof["B"] >= 0.5).astype(int)), 4))
    print("F_B tuned+pp :", round(macro_f1_techniques(Y_true, tech_pp), 4))
    print("F_C raw      :", round(macro_f1_multiclass(df["action"].values, oof["C"].argmax(1)), 4))
    print("F_C pp       :", round(macro_f1_multiclass(df["action"].values, act_pp), 4))


if __name__ == "__main__":
    main()
