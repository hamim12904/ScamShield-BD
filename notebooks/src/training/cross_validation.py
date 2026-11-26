"""Stage 3: leakage-safe unified folds, leave-cluster-out proxy, OOF saving, and slice scoring.

Run from the project root:
    python -m src.training.cross_validation
"""
import logging
import random
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

from src import config
from src.augment import corrupt as augment_corrupt
from src.data_loader import techniques_to_multihot
from src.preprocessing import preprocess, template_key
from src.text_analysis import language_bucket

logger = logging.getLogger(__name__)
FOLDS_PATH = config.VALIDATION_DIR / "folds.csv"
_HARD_LEGIT_RE = re.compile(r"\b(?:otp|link|parcel|cashback|urgent|bill)\b", re.I)


def corrupt_text(text, rng, strength=0.06, ops="B"):
    """Evaluation corruption using held-out Set B operators, preserving `|||`."""
    level = "light" if strength <= 0.045 else ("heavy" if strength >= 0.10 else "medium")
    return augment_corrupt(text, rng, level=level, ops=ops)


def assign_groups(df, sim_threshold=0.80):
    """Group exact templates, shared first messages (>=20 chars), and sparse cosine near-duplicates."""
    n = len(df)
    if n == 0:
        return pd.Series(dtype=np.int64, index=df.index, name="group")

    if "clean" in df.columns:
        cleans = df["clean"].fillna("").astype(str).tolist()
    else:
        cleans = [preprocess(t)["clean"] for t in df["text"]]

    if "template_key" in df.columns:
        keys = df["template_key"].fillna("").astype(str).tolist()
    else:
        keys = [template_key(c) for c in cleans]

    rows, cols = [], []

    # 1) Exact template_key matches
    by_key = {}
    for idx, key in enumerate(keys):
        by_key.setdefault(key, []).append(idx)
    for idxs in by_key.values():
        if len(idxs) > 1:
            base = idxs[0]
            for j in idxs[1:]:
                rows.extend((base, j))
                cols.extend((j, base))

    # 2) Shared first message (>= 20 chars after template_key)
    by_first = {}
    for idx, c in enumerate(cleans):
        first_msg = c.split(config.DELIMITER, 1)[0].strip()
        fkey = template_key(first_msg)
        if len(fkey) >= 20:
            by_first.setdefault(fkey, []).append(idx)
    for idxs in by_first.values():
        if len(idxs) > 1:
            base = idxs[0]
            for j in idxs[1:]:
                rows.extend((base, j))
                cols.extend((j, base))

    # 3) Sparse char n-gram near-duplicates
    nonempty = [i for i, c in enumerate(cleans) if c.strip()]
    if len(nonempty) > 1:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        X = vec.fit_transform([cleans[i] for i in nonempty])
        nn = NearestNeighbors(radius=max(0.0, 1.0 - sim_threshold), metric="cosine", algorithm="brute")
        nn.fit(X)
        graph = nn.radius_neighbors_graph(X, mode="connectivity").tocoo()
        for a, b in zip(graph.row, graph.col):
            if a < b:
                u, v = nonempty[int(a)], nonempty[int(b)]
                rows.extend((u, v))
                cols.extend((v, u))

    adj = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    return pd.Series(labels.astype(np.int64), index=df.index, name="group")


def assign_clusters(df, groups=None, n_clusters=3):
    """Per-label balanced template families using Ward linkage on L2-normalized char TF-IDF."""
    groups = np.asarray(assign_groups(df) if groups is None else groups)
    cleans = (
        df["clean"].fillna("").astype(str).tolist()
        if "clean" in df.columns
        else [preprocess(t)["clean"] for t in df["text"]]
    )
    out = np.zeros(len(df), dtype=np.int64)
    for lab in sorted(df["label"].unique()):
        idx = np.where(df["label"].values == lab)[0]
        ug = np.unique(groups[idx])
        k = min(n_clusters, len(ug))
        if k < 2:
            continue
        reps = [
            " ".join(cleans[i].strip() for i in idx if groups[i] == g and cleans[i].strip()) or "empty"
            for g in ug
        ]
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(reps).toarray()
        lab_k = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward").fit_predict(X)
        order = {c: r for r, c in enumerate(pd.Series(lab_k).value_counts().index)}  # 0 = largest family
        gmap = {g: order[c] for g, c in zip(ug, lab_k)}
        out[idx] = [gmap[groups[i]] for i in idx]
    return pd.Series(out, index=df.index, name="cluster")


def _resolve_groups(df, group_col="group", sim_threshold=0.80):
    if group_col in df.columns:
        return df[group_col]
    return assign_groups(df, sim_threshold=sim_threshold)


def make_folds(df, n_splits=config.N_SPLITS, seed=config.SEED, group_col="group"):
    """Unified StratifiedGroupKFold partition (stratified on Task A label, grouped by template)."""
    groups = _resolve_groups(df, group_col)
    grp_per_label = (
        pd.DataFrame({"label": df["label"].values, "group": np.asarray(groups)})
        .groupby("label")["group"]
        .nunique()
    )
    min_grp = int(grp_per_label.min())
    if min_grp < n_splits:
        new_splits = max(2, min_grp)
        logging.warning(
            "make_folds: smallest per-label group count (%d) < n_splits (%d); lowering to %d.",
            min_grp, n_splits, new_splits,
        )
        n_splits = new_splits
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from sgkf.split(df, df["label"], groups)


def _folds_to_array(n, folds):
    arr = np.full(n, -1, dtype=np.int64)
    for fold, (_, va) in enumerate(folds):
        arr[va] = fold
    assert (arr >= 0).all(), "Some rows were not assigned a validation fold."
    return arr


def build_fold_table(
    df,
    n_splits=config.N_SPLITS,
    n_repeats=config.N_REPEATS,
    seed=config.SEED,
    sim_threshold=0.80,
    n_clusters=3,
):
    """Build one unified repeated fold table across Tasks A/B/C plus per-label cluster IDs."""
    work = df.reset_index(drop=True).copy()
    work["group"] = assign_groups(work, sim_threshold=sim_threshold).values
    work["cluster"] = assign_clusters(work, work["group"], n_clusters=n_clusters).values

    cols = ["id", "group", "cluster"]
    for rep in range(n_repeats):
        col = f"fold_r{rep}"
        work[col] = _folds_to_array(len(work), make_folds(work, n_splits, seed + rep, "group"))
        cols.append(col)
    return work[cols]


def save_folds(df, path=FOLDS_PATH, **kwargs):
    """Build and persist unified fold assignments to CSV for reproducibility (Decision D-008)."""
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


def load_or_create_folds(
    df,
    path=FOLDS_PATH,
    n_splits=config.N_SPLITS,
    n_repeats=config.N_REPEATS,
    seed=config.SEED,
):
    """Read saved folds from disk (rebuilding with a warning only if missing or invalid)."""
    req = {"id", "group", "cluster"} | {f"fold_r{r}" for r in range(n_repeats)}
    if path.exists():
        table = pd.read_csv(path)
        if req.issubset(table.columns) and len(table) == len(df) and (table["id"].values == df["id"].values).all():
            return table
        logging.warning("load_or_create_folds: %s failed schema/ID check; rebuilding folds.", path)
    else:
        logging.warning("load_or_create_folds: %s missing; building folds.", path)
    return save_folds(df, path=path, n_splits=n_splits, n_repeats=n_repeats, seed=seed)


def iter_saved_folds(fold_table, col="fold_r0"):
    """Yield (train_idx, valid_idx) from a saved fold column."""
    vals = fold_table[col].values
    for f in sorted(np.unique(vals)):
        va = np.where(vals == f)[0]
        tr = np.where(vals != f)[0]
        yield tr, va


def get_folds(df, n_repeats=config.N_REPEATS):
    """Frozen contract: list of (train_idx, val_idx), ordered repeat then fold."""
    t = load_or_create_folds(df, n_repeats=n_repeats)
    return [p for r in range(n_repeats) for p in iter_saved_folds(t, f"fold_r{r}")]


def leave_cluster_out_folds(df, fold_table=None):
    """Yield (train_idx, valid_idx, cluster_id) holding out one template family per class."""
    if fold_table is not None and "cluster" in fold_table.columns:
        clusters = fold_table["cluster"].values
    elif "cluster" in df.columns:
        clusters = df["cluster"].values
    else:
        clusters = assign_clusters(df).values
    for cid in sorted(np.unique(clusters)):
        va = np.where(clusters == cid)[0]
        tr = np.where(clusters != cid)[0]
        if len(va) and len(tr):
            yield tr, va, int(cid)


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
    n_classes = getattr(config, "NUM_CLASSES", 6)
    n_actions = getattr(config, "NUM_ACTIONS", 6)
    n_tech = y_tech.shape[1]

    conf = np.asarray(pred["proba"]).max(axis=1)
    correct = (np.asarray(pred["label"]) == y_label).astype(float)
    brier = float(np.mean((conf - correct) ** 2))
    return {
        "F_A": _f1(y_label, pred["label"], list(range(n_classes))),
        "F_B": float(np.mean([
            f1_score(y_tech[:, j], np.asarray(pred["tech"])[:, j], zero_division=0)
            for j in range(n_tech)
        ])),
        "F_C": _f1(y_action, pred["action"], list(range(n_actions))),
        "brier": brier,
    }


def compute_slice_metrics(df, pred):
    """Evaluate Task A macro-F1 on language, conversation-structure, and hard-legit slices."""
    raw_texts = df["text"].fillna("").astype(str)
    langs = raw_texts.map(language_bucket).values
    n_msgs = (raw_texts.str.count(r"\|\|\|") + 1).values
    cleans = (
        df["clean"].fillna("").astype(str)
        if "clean" in df.columns
        else pd.Series([preprocess(t)["clean"] for t in raw_texts], index=df.index)
    )
    y = df["label"].values
    p = np.asarray(pred["label"])
    out = {}

    for lang in ("english", "banglish", "bangla_script", "mixed_bn_en"):
        mask = langs == lang
        if mask.any():
            present = sorted(np.unique(y[mask]).tolist())
            out[f"lang_{lang}"] = float(_f1(y[mask], p[mask], present))

    for name, mask in (("single", n_msgs == 1), ("conversation", n_msgs > 1)):
        if mask.any():
            present = sorted(np.unique(y[mask]).tolist())
            out[f"struct_{name}"] = float(_f1(y[mask], p[mask], present))

    hard_legit = np.isin(y, [0, 5]) & cleans.str.contains(_HARD_LEGIT_RE, na=False).values
    if hard_legit.any():
        present = sorted(np.unique(y[hard_legit]).tolist())
        out["slice_hard_legit"] = float(_f1(y[hard_legit], p[hard_legit], present))

    return out


def _init_oof(n, n_classes, n_tech, n_actions):
    """Allocate NaN/-1 sentinel OOF buffers so any unfilled row fails loudly."""
    return {
        "label": np.full(n, -1, dtype=np.int64),
        "proba": np.full((n, n_classes), np.nan, dtype=np.float64),
        "proba_b": np.full((n, n_tech), np.nan, dtype=np.float64),
        "tech": np.full((n, n_tech), -1, dtype=np.int64),
        "proba_c": np.full((n, n_actions), np.nan, dtype=np.float64),
        "action": np.full(n, -1, dtype=np.int64),
    }


def _assert_oof_complete(oof):
    assert (oof["label"] >= 0).all(), "Unfilled rows in OOF label."
    assert (oof["action"] >= 0).all(), "Unfilled rows in OOF action."
    assert (oof["tech"] >= 0).all(), "Unfilled rows in OOF tech."
    for k in ("proba", "proba_b", "proba_c"):
        assert not np.isnan(oof[k]).any(), f"NaN found in OOF {k}."


def save_oof_npz(
    df,
    oof_clean,
    oof_corrupted=None,
    folds_arr=None,
    n_repeats=config.N_REPEATS,
    exp_name="e001",
    suffix="m1",
):
    """Save repeat-averaged OOF probabilities to outputs/oof/<exp>_<suffix>.npz (D-008)."""
    out_path = config.OOF_DIR / f"{exp_name}_{suffix}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(df)
    y_a = df["label"].to_numpy(dtype=np.int64)
    y_b = techniques_to_multihot(df["techniques"]).astype(np.int64)
    y_c = df["action"].to_numpy(dtype=np.int64)
    prep = [preprocess(t) for t in df["text"]]
    noise_only = np.array([p["noise_only"] for p in prep], dtype=bool)
    noise_ratio = np.array([p["noise_ratio"] for p in prep], dtype=np.float64)

    p_a = np.asarray(oof_clean["proba"], dtype=np.float64)
    p_b = np.asarray(oof_clean["proba_b"], dtype=np.float64)
    p_c = np.asarray(oof_clean["proba_c"], dtype=np.float64)

    payload = {
        "ids": df["id"].to_numpy(dtype=np.int64),
        "y_a": y_a,
        "y_b": y_b,
        "y_c": y_c,
        "noise_only": noise_only,
        "noise_ratio": noise_ratio,
        "repeat": np.full(n, int(n_repeats), dtype=np.int64),
        # Shared contract for ensemble, calibration, and m2
        "A": p_a,
        "B": p_b,
        "C": p_c,
        # Backwards-compatible aliases
        "clean_proba_a": p_a,
        "clean_proba_b": p_b,
        "clean_proba_c": p_c,
    }
    if folds_arr is not None:
        payload["folds"] = np.asarray(folds_arr, dtype=np.int64)
    if oof_corrupted is not None:
        payload["corrupted_proba_a"] = np.asarray(oof_corrupted["proba"], dtype=np.float64)
        payload["corrupted_proba_b"] = np.asarray(oof_corrupted["proba_b"], dtype=np.float64)
        payload["corrupted_proba_c"] = np.asarray(oof_corrupted["proba_c"], dtype=np.float64)
    np.savez_compressed(out_path, **payload)
    return out_path


def cross_validate(
    df,
    make_model,
    n_splits=config.N_SPLITS,
    n_repeats=config.N_REPEATS,
    corrupt_valid=True,
    use_saved_folds=True,
    exp_name=None,
    model_tag="m1",
    return_oof=False,
):
    """Repeated grouped CV reading saved folds.csv, fitting once per fold, averaging OOF probs."""
    work = df.reset_index(drop=True).copy()
    if use_saved_folds:
        saved_folds = load_or_create_folds(work, FOLDS_PATH, n_splits=n_splits, n_repeats=n_repeats)
    else:
        saved_folds = build_fold_table(work, n_splits=n_splits, n_repeats=n_repeats)
    work["group"] = saved_folds["group"].values
    work["cluster"] = saved_folds["cluster"].values

    n = len(work)
    n_classes = getattr(config, "NUM_CLASSES", 6)
    n_actions = getattr(config, "NUM_ACTIONS", 6)
    n_tech = techniques_to_multihot(work["techniques"]).shape[1]

    rows = []
    slice_rows = []

    sum_clean = {
        "proba": np.zeros((n, n_classes), dtype=np.float64),
        "proba_b": np.zeros((n, n_tech), dtype=np.float64),
        "proba_c": np.zeros((n, n_actions), dtype=np.float64),
    }
    sum_noisy = {
        "proba": np.zeros((n, n_classes), dtype=np.float64),
        "proba_b": np.zeros((n, n_tech), dtype=np.float64),
        "proba_c": np.zeros((n, n_actions), dtype=np.float64),
    }

    for rep in range(n_repeats):
        rng = random.Random(config.SEED + rep)
        fold_iter = list(iter_saved_folds(saved_folds, f"fold_r{rep}"))

        oof_clean = _init_oof(n, n_classes, n_tech, n_actions)
        oof_noisy = _init_oof(n, n_classes, n_tech, n_actions)

        for tr, va in fold_iter:
            train, valid = work.iloc[tr], work.iloc[va].copy()
            model = make_model()
            model.fit(train)

            pred = model.predict(valid)
            for k in oof_clean:
                if k in pred:
                    oof_clean[k][va] = pred[k]

            if corrupt_valid:
                noisy = valid.copy()
                noisy["text"] = noisy["text"].map(lambda x: corrupt_text(x, rng, ops="B"))
                pred_n = model.predict(noisy)
                for k in oof_noisy:
                    if k in pred_n:
                        oof_noisy[k][va] = pred_n[k]

        _assert_oof_complete(oof_clean)
        rows.append({"rep": rep, "view": "clean", **score(work, oof_clean)})
        slice_rows.append({"rep": rep, "view": "clean", **compute_slice_metrics(work, oof_clean)})
        for k in sum_clean:
            sum_clean[k] += oof_clean[k]

        if corrupt_valid:
            _assert_oof_complete(oof_noisy)
            rows.append({"rep": rep, "view": "corrupted", **score(work, oof_noisy)})
            slice_rows.append({"rep": rep, "view": "corrupted", **compute_slice_metrics(work, oof_noisy)})
            for k in sum_noisy:
                sum_noisy[k] += oof_noisy[k]

    def _avg_oof(sums):
        pa = sums["proba"] / n_repeats
        pb = sums["proba_b"] / n_repeats
        pc = sums["proba_c"] / n_repeats
        return {
            "proba": pa,
            "proba_b": pb,
            "proba_c": pc,
            "label": pa.argmax(axis=1),
            "tech": (pb >= 0.5).astype(np.int64),
            "action": pc.argmax(axis=1),
        }

    avg_clean = _avg_oof(sum_clean)
    avg_corrupted = _avg_oof(sum_noisy) if corrupt_valid else None

    if exp_name is not None:
        save_oof_npz(
            work,
            avg_clean,
            avg_corrupted,
            folds_arr=saved_folds["fold_r0"].values,
            n_repeats=n_repeats,
            exp_name=exp_name,
            suffix=model_tag,
        )

    res = pd.DataFrame(rows)
    res_slices = pd.DataFrame(slice_rows)
    summary = res.groupby("view")[["F_A", "F_B", "F_C", "brier"]].agg(["mean", "std"]).round(3)
    if return_oof:
        return res, summary, {
            "clean": avg_clean,
            "corrupted": avg_corrupted,
            "slices": res_slices,
            "folds": saved_folds,
        }
    return res, summary


def leave_cluster_out_cv(df, make_model, saved_folds=None):
    """Domain-shift / novel-scenario proxy by holding out one template family per class."""
    work = df.reset_index(drop=True).copy()
    if saved_folds is None:
        saved_folds = load_or_create_folds(work)
    work["group"] = saved_folds["group"].values
    work["cluster"] = saved_folds["cluster"].values

    n = len(work)
    n_classes = getattr(config, "NUM_CLASSES", 6)
    n_actions = getattr(config, "NUM_ACTIONS", 6)
    n_tech = techniques_to_multihot(work["techniques"]).shape[1]
    oof = _init_oof(n, n_classes, n_tech, n_actions)

    for tr, va, _ in leave_cluster_out_folds(work, saved_folds):
        model = make_model()
        model.fit(work.iloc[tr])
        pred = model.predict(work.iloc[va])
        for k in oof:
            if k in pred:
                oof[k][va] = pred[k]

    _assert_oof_complete(oof)
    return score(work, oof), oof


if __name__ == "__main__":
    from src.data_loader import load_sample
    df = load_sample()
    folds = save_folds(df)
    print(f"Saved {len(folds)} unified fold assignments to {FOLDS_PATH}")
    print("Unique duplicate/near-duplicate groups:", folds["group"].nunique(), "of", len(folds))
    print("\nCluster x Label cross-tab (every class should span >=2 clusters):")
    ct = pd.crosstab(folds["cluster"], df["label"])
    print(ct)

    Y_tech = techniques_to_multihot(df["techniques"])
    for rep in range(config.N_REPEATS):
        col = f"fold_r{rep}"
        assert (folds.groupby("group")[col].nunique() > 1).sum() == 0, f"Group leakage in {col}"
        if rep == 0:
            print(f"\n--- Unified partition {col} (Tasks A / B / C) ---")
            for f, (_, va) in enumerate(iter_saved_folds(folds, col)):
                t_cnt = {config.TECHNIQUES[j]: int(Y_tech[va, j].sum()) for j in range(6)}
                print(
                    f"  fold {f}: n={len(va):2d} | A={df.iloc[va].label.value_counts().sort_index().to_dict()} "
                    f"| C={df.iloc[va].action.value_counts().sort_index().to_dict()} | B={t_cnt}"
                )

    assert (ct > 0).sum(axis=0).min() >= 2, "A class appears in fewer than 2 clusters!"
    print("\nGroup leakage and LCO cluster checks passed.")
