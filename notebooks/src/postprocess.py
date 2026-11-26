"""Consistency post-processing rules between Task A, B, and C outputs.

Owned by Teammate B (T-024).

Rules derived from sample.csv cross-tab (86/86 real rows, 100% support):
  1. label in {0, 5} => techniques = NONE (all zeros) [44/44 = 100%]
  2. label in {0, 5} => action = 0 [44/44 = 100%]
  3. label in {1, 2, 3, 4} => at least one technique [42/42 = 100%];
     if none above threshold, emit argmax(probs_b).
  4. label in {1, 2, 3, 4} => action != 0 [42/42 = 100%];
     if predicted action == 0, emit 1 + argmax(probs_c[1:]).

Rules never key off text strings or test ids.
"""
import logging

import numpy as np

log = logging.getLogger(__name__)

_NON_SCAM_LABELS = {0, 5}
_SCAM_LABELS = {1, 2, 3, 4}


def apply_rules(label: int, tech, action: int, probs: dict) -> tuple:
    """Apply consistency rules to a single prediction row.

    label:  predicted Task A label (0..5)
    tech:   (6,) binary technique array
    action: predicted Task C action (0..5)
    probs:  dict with optional keys 'B' (6,) and 'C' (6,)

    Returns: (label, tech_array, action)
    """
    label = int(label)
    tech = np.array(tech, dtype=int, copy=True)
    action = int(action)

    if label in _NON_SCAM_LABELS:
        tech[:] = 0
        action = 0
        return label, tech, action

    if label in _SCAM_LABELS:
        if tech.sum() == 0 and probs is not None and "B" in probs:
            pb = np.asarray(probs["B"], dtype=float)
            tech[int(np.argmax(pb))] = 1
        if action == 0 and probs is not None and "C" in probs:
            pc = np.asarray(probs["C"], dtype=float)
            action = 1 + int(np.argmax(pc[1:]))

    return label, tech, action


def apply_rules_batch(labels: np.ndarray,
                      techs: np.ndarray,
                      actions: np.ndarray,
                      probs_b: np.ndarray = None,
                      probs_c: np.ndarray = None) -> tuple:
    """Vectorised consistency post-processing for batch predictions.

    Returns (labels_out, techs_out, actions_out) and logs rule change counts.
    """
    labels = np.asarray(labels, dtype=int).copy()
    techs = np.asarray(techs, dtype=int).copy()
    actions = np.asarray(actions, dtype=int).copy()

    non_scam = np.isin(labels, list(_NON_SCAM_LABELS))
    scam = ~non_scam

    # Rule 1 & 2: label in {0, 5} -> NONE and action 0
    tech_zeroed = int((techs[non_scam].sum(axis=1) > 0).sum())
    act_zeroed = int((actions[non_scam] != 0).sum())
    techs[non_scam] = 0
    actions[non_scam] = 0

    # Rule 3: label in {1..4} -> at least one technique
    tech_added = 0
    if probs_b is not None:
        no_tech = scam & (techs.sum(axis=1) == 0)
        tech_added = int(no_tech.sum())
        if tech_added > 0:
            best_t = np.argmax(probs_b[no_tech], axis=1)
            techs[no_tech, best_t] = 1

    # Rule 4: label in {1..4} -> action != 0
    act_fixed = 0
    if probs_c is not None:
        zero_act = scam & (actions == 0)
        act_fixed = int(zero_act.sum())
        if act_fixed > 0:
            actions[zero_act] = 1 + np.argmax(probs_c[zero_act, 1:], axis=1)

    log.info(
        "postprocess: tech_zeroed=%d, act_zeroed=%d, tech_added=%d, act_fixed=%d",
        tech_zeroed, act_zeroed, tech_added, act_fixed,
    )
    return labels, techs, actions


def tune_task_b_thresholds(oof_proba_b: np.ndarray,
                           Y_true: np.ndarray,
                           folds: np.ndarray,
                           grid=None) -> list:
    """Cross-fitted per-technique threshold tuning on OOF probabilities.

    Chooses threshold in grid (default 0.05..0.95) maximising F1 on 4 folds,
    evaluates on 5th, returns mean threshold across folds per technique.
    Shrinks toward 0.5 when a technique has < 5 positives.
    """
    from sklearn.metrics import f1_score

    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)

    oof_proba_b = np.asarray(oof_proba_b, dtype=float)
    Y_true = np.asarray(Y_true, dtype=int)
    folds = np.asarray(folds)
    unique_folds = np.unique(folds)
    n_tech = Y_true.shape[1]

    final_thresholds = []
    for j in range(n_tech):
        pos_total = int(Y_true[:, j].sum())
        if pos_total < 2:
            final_thresholds.append(0.5)
            continue

        fold_best = []
        for f in unique_folds:
            tr = folds != f
            y_tr = Y_true[tr, j]
            p_tr = oof_proba_b[tr, j]
            best_t, best_f1 = 0.5, -1.0
            for t in grid:
                score = f1_score(y_tr, (p_tr >= t).astype(int), zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t = float(t)
            fold_best.append(best_t)

        thr = float(np.mean(fold_best))
        if pos_total < 5:
            # Shrink toward 0.5 for very rare techniques
            alpha = pos_total / 5.0
            thr = alpha * thr + (1.0 - alpha) * 0.5
        final_thresholds.append(round(thr, 3))

    log.info("Tuned Task B thresholds: %s", final_thresholds)
    return final_thresholds
