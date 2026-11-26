"""Stage 3: leakage-safe folds, corruption for robustness checks, metrics.

fit_predict(train_df, valid_df) must return a dict with:
    "label": int array (n,), "proba": float array (n, 6),
    "action": int array (n,), "tech": int array (n, 6)
"""
import random

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src.data_loader import techniques_to_multihot
from src.preprocessing import _LOOKALIKE_PAIRS

_REVERSE = {}
for k, v in _LOOKALIKE_PAIRS.items():
    _REVERSE.setdefault(v, []).append(k)
EMOJIS = ["😊", "🎁", "🔥", "💰", "✅", "📱", "🙏"]


def corrupt_text(text, rng, strength=0.06):
    """Realistic distortions from Section 7 of the problem statement."""
    out = []
    for ch in str(text):
        r = rng.random()
        if r < strength * 0.25:                       # deletion
            continue
        if r < strength * 0.5:                        # repetition
            out.append(ch * rng.randint(2, 3))
            continue
        if r < strength * 0.75 and ch in _REVERSE:    # lookalike substitution
            out.append(rng.choice(_REVERSE[ch]))
            continue
        if r < strength * 0.85:                       # zero-width insertion
            out.append(ch + "\u200b")
            continue
        if r < strength:                              # random case
            out.append(ch.swapcase())
            continue
        out.append(ch)
    t = "".join(out)
    if rng.random() < 0.3:
        t = t.replace(" ", "", rng.randint(1, 3))     # missing spaces
    if rng.random() < 0.3:
        t = f"{t} {rng.choice(EMOJIS)}"
    if rng.random() < 0.2:
        noise = "".join(rng.choice("bcdfghjklmnpqrstvwxz") for _ in range(rng.randint(8, 11)))
        words = t.split(" ")
        words.insert(rng.randint(0, len(words)), noise)
        t = " ".join(words)
    return t


def make_folds(df, n_splits=5, seed=config.SEED, group_col="group"):
    """Yield (train_idx, valid_idx). Stratified on label, grouped by template/source."""
    groups = df[group_col] if group_col in df.columns else df["id"]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from sgkf.split(df, df["label"], groups)


def language_holdout_folds(df, language_col="language"):
    """Domain-shift check: train on two language buckets, validate on the third."""
    for lang in df[language_col].unique():
        va = np.where(df[language_col] == lang)[0]
        tr = np.where(df[language_col] != lang)[0]
        if len(va) and len(tr):
            yield tr, va, lang


def _f1(y, p, labels):
    return f1_score(y, p, labels=labels, average="macro", zero_division=0)


def score(valid, pred):
    y_label = valid["label"].values
    y_action = valid["action"].values
    y_tech = techniques_to_multihot(valid["techniques"])
    conf = np.asarray(pred["proba"]).max(axis=1)
    correct = (np.asarray(pred["label"]) == y_label).astype(float)
    brier = float(np.mean((conf - correct) ** 2))
    return {
        "F_A": _f1(y_label, pred["label"], list(range(6))),
        "F_B": float(np.mean([
            f1_score(y_tech[:, j], np.asarray(pred["tech"])[:, j], zero_division=0)
            for j in range(6)])),
        "F_C": _f1(y_action, pred["action"], list(range(6))),
        "brier": brier,
    }


def cross_validate(df, fit_predict, n_splits=5, n_repeats=3, corrupt_valid=True):
    """Repeated grouped CV on clean and corrupted validation text."""
    rows = []
    for rep in range(n_repeats):
        seed = config.SEED + rep
        rng = random.Random(seed)
        for fold, (tr, va) in enumerate(make_folds(df, n_splits, seed)):
            train, valid = df.iloc[tr], df.iloc[va].copy()
            pred = fit_predict(train, valid)
            rows.append({"rep": rep, "fold": fold, "view": "clean", **score(valid, pred)})
            if corrupt_valid:
                noisy = valid.copy()
                noisy["text"] = noisy["text"].map(lambda x: corrupt_text(x, rng))
                pred_n = fit_predict(train, noisy)
                rows.append({"rep": rep, "fold": fold, "view": "corrupted", **score(valid, pred_n)})
    res = pd.DataFrame(rows)
    summary = res.groupby("view")[["F_A", "F_B", "F_C", "brier"]].agg(["mean", "std"]).round(3)
    return res, summary


if __name__ == "__main__":
    from src.data_loader import load_sample
    df = load_sample()
    df["group"] = df["id"]
    for i, (tr, va) in enumerate(make_folds(df)):
        print(f"fold {i}: train={len(tr)} valid={len(va)} "
              f"valid label counts={df.iloc[va].label.value_counts().sort_index().to_dict()}")
    rng = random.Random(config.SEED)
    for text in df.text.head(4):
        print("\nORIG :", text[:100])
        print("CORR :", corrupt_text(text, rng)[:100])