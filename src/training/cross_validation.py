"""Stage 3: leakage-safe folds, corruption for robustness checks, metrics.

Run from the project root:
    python -m src.training.cross_validation

fit_predict(train_df, valid_df) must return a dict with:
    "label": int array (n,), "proba": float array (n, 6),
    "action": int array (n,), "tech": int array (n, 6)
"""
import random

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src.data_loader import techniques_to_multihot
from src.preprocessing import _LOOKALIKE_PAIRS, preprocess, template_key

_REVERSE = {}
for k, v in _LOOKALIKE_PAIRS.items():
    _REVERSE.setdefault(v, []).append(k)
EMOJIS = ["😊", "🎁", "🔥", "💰", "✅", "📱", "🙏"]
FOLDS_PATH = config.VALIDATION_DIR / "folds.csv"


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


def assign_groups(df, sim_threshold=0.80):
    """Group exact templates and char-TF-IDF near-duplicates so they never split across folds."""
    n = len(df)
    if "clean" in df.columns:
        cleans = df["clean"].fillna("").astype(str).tolist()
    else:
        cleans = [preprocess(t)["clean"] for t in df["text"]]

    if "template_key" in df.columns:
        keys = df["template_key"].fillna("").astype(str).tolist()
    else:
        keys = [template_key(c) for c in cleans]

    # Build adjacency from shared template_key and near-duplicate char n-gram cosine similarity
    adj = np.zeros((n, n), dtype=bool)
    by_key = {}
    for idx, key in enumerate(keys):
        by_key.setdefault(key, []).append(idx)
    for idxs in by_key.values():
        if len(idxs) > 1:
            base = idxs[0]
            for j in idxs[1:]:
                adj[base, j] = adj[j, base] = True

    nonempty = [i for i, c in enumerate(cleans) if c.strip()]
    if len(nonempty) > 1:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        X = vec.fit_transform([cleans[i] for i in nonempty])
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, 0)
        ii, jj = np.where(np.triu(sim) >= sim_threshold)
        for a, b in zip(ii, jj):
            u, v = nonempty[a], nonempty[b]
            adj[u, v] = adj[v, u] = True

    _, labels = connected_components(adj, directed=False)
    return pd.Series(labels.astype(np.int64), index=df.index, name="group")


def _resolve_groups(df, group_col="group", sim_threshold=0.80):
    if group_col in df.columns:
        return df[group_col]
    return assign_groups(df, sim_threshold=sim_threshold)


def make_folds(df, n_splits=5, seed=config.SEED, group_col="group"):
    """Task A folds: stratified on `label`, grouped by duplicate/near-duplicate template."""
    groups = _resolve_groups(df, group_col)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from sgkf.split(df, df["label"], groups)


def make_task_a_folds(df, n_splits=5, seed=config.SEED, group_col="group"):
    """Alias for Task A stratified grouped folds."""
    yield from make_folds(df, n_splits=n_splits, seed=seed, group_col=group_col)


def make_task_c_folds(df, n_splits=3, seed=config.SEED, group_col="group"):
    """Task C folds for imbalanced `action` (caps n_splits at smallest action group count)."""
    groups = _resolve_groups(df, group_col)
    grp_per_action = (
        pd.DataFrame({"action": df["action"].values, "group": groups.values})
        .groupby("action")["group"]
        .nunique()
    )
    feasible_splits = max(2, min(int(n_splits), int(grp_per_action.min())))
    sgkf = StratifiedGroupKFold(n_splits=feasible_splits, shuffle=True, random_state=seed)
    yield from sgkf.split(df, df["action"], groups)


def make_task_b_folds(df, n_splits=5, seed=config.SEED, group_col="group"):
    """Task B multilabel-aware grouped folds using greedy iterative stratification."""
    groups = np.asarray(_resolve_groups(df, group_col))
    Y_raw = techniques_to_multihot(df["techniques"])
    # Include an explicit NONE indicator so legitimate/promo records also balance evenly
    is_none = (Y_raw.sum(axis=1) == 0).astype(int)[:, None]
    Y = np.hstack([Y_raw, is_none])

    uniq_groups = np.unique(groups)
    grp_idx = {g: np.where(groups == g)[0] for g in uniq_groups}
    grp_vec = {g: Y[idxs].sum(axis=0) for g, idxs in grp_idx.items()}
    grp_size = {g: len(idxs) for g, idxs in grp_idx.items()}

    rng = random.Random(seed)
    desired = np.tile(Y.sum(axis=0) / n_splits, (n_splits, 1))
    fold_sizes = np.zeros(n_splits, dtype=int)
    remaining = set(uniq_groups.tolist())
    fold_of_group = {}

    while remaining:
        rem_list = sorted(remaining)
        rem_mat = np.array([grp_vec[g] for g in rem_list])
        col_totals = rem_mat.sum(axis=0)
        pos_cols = np.where(col_totals > 0)[0]

        if len(pos_cols) > 0:
            min_total = col_totals[pos_cols].min()
            cand_cols = pos_cols[col_totals[pos_cols] == min_total]
            target_col = int(rng.choice(cand_cols.tolist()))
            cand_groups = [g for g in rem_list if grp_vec[g][target_col] > 0]
            rng.shuffle(cand_groups)
            g_pick = max(cand_groups, key=lambda g: grp_vec[g][target_col])
            des_vals = desired[:, target_col]
        else:
            rng.shuffle(rem_list)
            g_pick = max(rem_list, key=lambda g: grp_size[g])
            des_vals = np.full(n_splits, -fold_sizes.astype(float))

        max_des = des_vals.max()
        best_folds = [f for f in range(n_splits) if np.isclose(des_vals[f], max_des)]
        min_sz = min(fold_sizes[f] for f in best_folds)
        best_folds = [f for f in best_folds if fold_sizes[f] == min_sz]
        f_pick = rng.choice(best_folds)

        fold_of_group[g_pick] = f_pick
        desired[f_pick] -= grp_vec[g_pick]
        fold_sizes[f_pick] += grp_size[g_pick]
        remaining.remove(g_pick)

    row_folds = np.array([fold_of_group[g] for g in groups], dtype=int)
    for f in range(n_splits):
        va = np.where(row_folds == f)[0]
        tr = np.where(row_folds != f)[0]
        yield tr, va


def _folds_to_array(n, folds):
    arr = np.full(n, -1, dtype=np.int64)
    for fold, (_, va) in enumerate(folds):
        arr[va] = fold
    assert (arr >= 0).all(), "Some rows were not assigned a validation fold."
    return arr


def build_fold_table(df, n_splits_a=5, n_splits_b=5, n_splits_c=3, seed=config.SEED, sim_threshold=0.80):
    """Compute group IDs and Task A / B / C fold assignments for every row."""
    work = df.copy()
    work["group"] = assign_groups(work, sim_threshold=sim_threshold)
    work["task_a_fold"] = _folds_to_array(len(work), make_task_a_folds(work, n_splits_a, seed, "group"))
    work["task_b_fold"] = _folds_to_array(len(work), make_task_b_folds(work, n_splits_b, seed, "group"))
    work["task_c_fold"] = _folds_to_array(len(work), make_task_c_folds(work, n_splits_c, seed, "group"))
    return work[["id", "group", "task_a_fold", "task_b_fold", "task_c_fold"]]


def save_folds(df, path=FOLDS_PATH, **kwargs):
    """Build and persist validation fold assignments to CSV for reproducibility."""
    table = build_fold_table(df, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def load_folds(path=FOLDS_PATH, df=None):
    """Load saved fold assignments and verify row/ID alignment if `df` is provided."""
    table = pd.read_csv(path)
    if df is not None:
        assert len(table) == len(df), f"Fold table length {len(table)} != df length {len(df)}"
        assert (table["id"].values == df["id"].values).all(), "Fold table IDs do not match df IDs."
    return table


def iter_saved_folds(fold_table, col="task_a_fold"):
    """Yield (train_idx, valid_idx) from a saved fold column."""
    vals = fold_table[col].values
    for f in sorted(np.unique(vals)):
        va = np.where(vals == f)[0]
        tr = np.where(vals != f)[0]
        yield tr, va


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
    work = df.copy()
    if "group" not in work.columns:
        work["group"] = assign_groups(work)
    rows = []
    for rep in range(n_repeats):
        seed = config.SEED + rep
        rng = random.Random(seed)
        for fold, (tr, va) in enumerate(make_folds(work, n_splits, seed)):
            train, valid = work.iloc[tr], work.iloc[va].copy()
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
    folds = save_folds(df)
    print(f"Saved {len(folds)} fold assignments to {FOLDS_PATH}")
    print("Unique duplicate/near-duplicate groups:", folds["group"].nunique(), "of", len(folds))

    print("\n--- Task A folds (stratified grouped, 5-fold) ---")
    for f, (_, va) in enumerate(iter_saved_folds(folds, "task_a_fold")):
        print(f"  fold {f}: n={len(va):2d} labels={df.iloc[va].label.value_counts().sort_index().to_dict()}")

    print("\n--- Task B folds (multilabel-aware grouped, 5-fold) ---")
    Y = techniques_to_multihot(df["techniques"])
    for f, (_, va) in enumerate(iter_saved_folds(folds, "task_b_fold")):
        t_counts = {config.TECHNIQUES[j]: int(Y[va, j].sum()) for j in range(6)}
        n_none = int((Y[va].sum(axis=1) == 0).sum())
        print(f"  fold {f}: n={len(va):2d} NONE={n_none:2d} techs={t_counts}")

    print("\n--- Task C folds (imbalanced action grouped, 3-fold) ---")
    for f, (_, va) in enumerate(iter_saved_folds(folds, "task_c_fold")):
        print(f"  fold {f}: n={len(va):2d} actions={df.iloc[va].action.value_counts().sort_index().to_dict()}")

    # Verify group integrity across every fold column
    for col in ("task_a_fold", "task_b_fold", "task_c_fold"):
        n_split_groups = (folds.groupby("group")[col].nunique() > 1).sum()
        assert n_split_groups == 0, f"Leakage in {col}: {n_split_groups} groups split!"
    print("\nGroup leakage check passed: 0 groups split across train/valid in any task.")