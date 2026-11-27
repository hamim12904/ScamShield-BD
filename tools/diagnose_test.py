"""Aggregate-only diagnosis of the E021 class-0 test skew.

CLI:
    python tools/diagnose_test.py

Writes outputs/validation_results/test_diagnosis.md. Library code logs and never
emits individual test texts or ids.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.data_loader import load_sample, load_test, parse_techniques
from src.preprocessing import URL_RE
from src.text_analysis import language_bucket
from src.training.train_baseline import (
    _CRED_RE,
    _MONEY_RE,
    _REWARD_RE,
    _URGENCY_RE,
    load_or_build_preprocessed_test,
)

log = logging.getLogger(__name__)

OTP_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")
LANGUAGE_ORDER = ("english", "banglish", "bangla_script", "mixed_bn_en", "no_letters")
CUE_COLUMNS = (
    "noise_only", "has_url", "otp_like", "money", "cred",
    "urgency", "reward", "conversation",
)


def _pct(count: int, total: int) -> float:
    """Return count as a percentage of total, or 0.0 when total is zero."""
    return 100.0 * float(count) / float(total) if total else 0.0


def m2_coverage(a: np.ndarray, ood_sim: Optional[np.ndarray] = None) -> Dict:
    """Count M2 rows matching the padded tail vector and its constant ood_sim tail."""
    probs = np.asarray(a, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] == 0:
        raise ValueError("M2 probabilities must be a non-empty 2-D array")
    padded_mask = np.all(probs == probs[-1], axis=1)
    padded = int(padded_mask.sum())
    total = int(len(probs))
    out = {
        "n": total,
        "padded": padded,
        "real": total - padded,
        "padded_pct": _pct(padded, total),
        "real_pct": _pct(total - padded, total),
    }
    if ood_sim is not None:
        scores = np.asarray(ood_sim, dtype=np.float64).reshape(-1)
        if len(scores) != total:
            raise ValueError("ood_sim length must match probability rows")
        ood_tail_mask = scores == scores[-1]
        tail_unique = int(np.unique(scores[padded_mask]).size) if padded else 0
        out["ood_equal_last"] = int(ood_tail_mask.sum())
        out["ood_equal_last_pct"] = _pct(int(ood_tail_mask.sum()), total)
        out["ood_tail_unique"] = tail_unique
        out["ood_constant_tail"] = bool(padded > 0 and tail_unique == 1)
    return out


def histograms(labels: Sequence[int], actions: Sequence[int],
               techniques: Sequence[str]) -> Dict:
    """Return aggregate label, action, and technique histograms."""
    label_counts = Counter(int(v) for v in labels)
    action_counts = Counter(int(v) for v in actions)
    tech_counts: Counter = Counter()
    for value in techniques:
        codes = parse_techniques(value)
        if codes:
            tech_counts.update(codes)
        else:
            tech_counts["NONE"] += 1
    return {
        "label": {k: int(label_counts.get(k, 0)) for k in range(config.NUM_CLASSES)},
        "action": {k: int(action_counts.get(k, 0)) for k in range(config.NUM_ACTIONS)},
        "techniques": {k: int(tech_counts.get(k, 0)) for k in ["NONE", *config.TECHNIQUES]},
    }


def cue_rates(prep: pd.DataFrame, labels: Sequence[int]) -> Dict:
    """Return cue counts and rates overall, for predicted 0, and for predicted 1-4."""
    frame = prep.reset_index(drop=True)
    lab = pd.Series(np.asarray(labels), dtype="int64")
    if len(lab) != len(frame):
        raise ValueError("labels length must match preprocessed rows")

    text = frame["preserved"].fillna("").astype(str)
    flags = pd.DataFrame(index=frame.index)
    flags["noise_only"] = frame["noise_only"].fillna(False).astype(bool)
    if "has_url" in frame.columns:
        flags["has_url"] = frame["has_url"].fillna(False).astype(bool)
    else:
        flags["has_url"] = text.map(
            lambda s: bool(URL_RE.search(s)) if ("." in s or "://" in s) else False
        )
    flags["otp_like"] = text.map(lambda s: bool(OTP_RE.search(s)))
    flags["money"] = text.map(lambda s: bool(_MONEY_RE.search(s)))
    flags["cred"] = text.map(lambda s: bool(_CRED_RE.search(s)))
    flags["urgency"] = text.map(lambda s: bool(_URGENCY_RE.search(s)))
    flags["reward"] = text.map(lambda s: bool(_REWARD_RE.search(s)))
    flags["conversation"] = frame["n_messages"].fillna(1).astype(int) > 1

    groups = {
        "all": pd.Series(True, index=frame.index),
        "pred_0": lab == 0,
        "pred_1_4": lab.isin([1, 2, 3, 4]),
    }
    out = {}
    for name, mask in groups.items():
        mask = mask.reset_index(drop=True)
        n = int(mask.sum())
        out[name] = {"n": n}
        for col in CUE_COLUMNS:
            count = int(flags.loc[mask, col].sum())
            out[name][col] = {"count": count, "pct": _pct(count, n)}
    return out


def _conf_stats(values: pd.Series) -> Dict:
    """Return aggregate confidence mean and 10/50/90 quantiles."""
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "q10": float(values.quantile(0.10)),
        "q50": float(values.quantile(0.50)),
        "q90": float(values.quantile(0.90)),
    }


def confidence_stats(labels: Sequence[int], confidence: Sequence[float]) -> Dict:
    """Return confidence summary for all rows and each predicted class."""
    frame = pd.DataFrame({
        "label": np.asarray(labels, dtype=np.int64),
        "confidence": np.asarray(confidence, dtype=np.float64),
    })
    out = {"all": _conf_stats(frame["confidence"])}
    for label in sorted(frame["label"].unique().tolist()):
        out[str(int(label))] = _conf_stats(frame.loc[frame["label"] == label, "confidence"])
    return out


def language_mix(sample_texts: Sequence[str], test_texts: Sequence[str],
                 labels: Sequence[int]) -> Dict:
    """Return sample/test language shares and predicted class mix per test bucket."""
    sample_buckets = pd.Series(list(sample_texts), dtype="string").fillna("").map(language_bucket)
    test_buckets = pd.Series(list(test_texts), dtype="string").fillna("").map(language_bucket)
    lab = np.asarray(labels, dtype=np.int64)
    if len(lab) != len(test_buckets):
        raise ValueError("labels length must match test rows")

    def shares(series: pd.Series) -> Dict:
        counts = series.value_counts()
        return {
            bucket: {"count": int(counts.get(bucket, 0)), "pct": _pct(int(counts.get(bucket, 0)), len(series))}
            for bucket in LANGUAGE_ORDER
        }

    by_bucket = {}
    for bucket in LANGUAGE_ORDER:
        mask = (test_buckets == bucket).to_numpy()
        n = int(mask.sum())
        counts = Counter(lab[mask].tolist())
        by_bucket[bucket] = {
            "n": n,
            "pct": _pct(n, len(test_buckets)),
            "label_counts": {str(k): int(counts.get(k, 0)) for k in range(config.NUM_CLASSES)},
        }
    return {
        "sample": shares(sample_buckets),
        "test": shares(test_buckets),
        "pred_by_bucket": by_bucket,
    }


def m1_m2_disagreement(m1_a: Optional[np.ndarray] = None,
                       m2_a: Optional[np.ndarray] = None,
                       padded_mask: Optional[np.ndarray] = None) -> Dict:
    """Compare M1/M2 argmax on non-padded rows only, or report unavailable input."""
    if m1_a is None or m2_a is None:
        return {"available": False, "reason": "M1 test probabilities are not saved"}
    p1 = np.asarray(m1_a, dtype=np.float64)
    p2 = np.asarray(m2_a, dtype=np.float64)
    if p1.shape != p2.shape:
        raise ValueError("M1 and M2 probability arrays must have matching shapes")
    mask = np.ones(len(p2), dtype=bool) if padded_mask is None else ~np.asarray(padded_mask, dtype=bool)
    if len(mask) != len(p2):
        raise ValueError("padded_mask length must match M2 rows")
    disagreements = int((p1[mask].argmax(axis=1) != p2[mask].argmax(axis=1)).sum())
    n = int(mask.sum())
    return {"available": True, "n": n, "disagreements": disagreements, "pct": _pct(disagreements, n)}


def _fmt_count_pct(item: Mapping) -> str:
    """Format a count/percentage aggregate pair."""
    return f"{int(item['count']):,} ({item['pct']:.2f}%)"


def render_report(diagnostics: Mapping) -> str:
    """Render aggregate-only diagnosis markdown from pure diagnostic dictionaries."""
    coverage = diagnostics["coverage"]
    sample_hist = diagnostics["sample_hist"]
    pred_hist = diagnostics["pred_hist"]
    cue = diagnostics["cue"]
    conf = diagnostics["conf"]
    lang = diagnostics["lang"]
    disagree = diagnostics["disagree"]

    lines = [
        "# Test-set class-0 skew diagnosis",
        "",
        "Aggregate counts and percentages only. All test labels below are model predictions; "
        "no test ground truth exists. No individual test text or id is included.",
        "",
        "## 1. Model 2 coverage (H1)",
        "",
        f"- M2 probability rows: {coverage['n']:,}",
        f"- Rows equal to the padded tail vector: **{coverage['padded']:,} ({coverage['padded_pct']:.2f}%)**",
        f"- Rows with a distinct M2 vector: {coverage['real']:,} ({coverage['real_pct']:.2f}%)",
    ]
    if "ood_constant_tail" in coverage:
        lines.append(
            f"- ood_sim values equal to the tail value: {coverage['ood_equal_last']:,} "
            f"({coverage['ood_equal_last_pct']:.2f}%); distinct values in padded tail: "
            f"{coverage['ood_tail_unique']}"
        )
        lines.append(f"- Constant ood_sim tail: {'yes' if coverage['ood_constant_tail'] else 'no'}")
    lines.extend([
        f"- M2 covers all test rows: {'yes' if coverage['padded'] == 0 else 'no'}",
        "",
        "## 2. Predicted histograms vs labeled sample",
        "",
        "| Class | Sample count | Sample % | Predicted count | Predicted % |",
        "|---|---:|---:|---:|---:|",
    ])
    sample_total = sum(sample_hist["label"].values())
    pred_total = sum(pred_hist["label"].values())
    for label in range(config.NUM_CLASSES):
        sc = sample_hist["label"][label]
        pc = pred_hist["label"][label]
        lines.append(f"| {label} | {sc:,} | {_pct(sc, sample_total):.2f}% | {pc:,} | {_pct(pc, pred_total):.2f}% |")

    lines.extend([
        "",
        "Predicted actions:",
        "",
        "| Action | Count | % |",
        "|---|---:|---:|",
    ])
    for action in range(config.NUM_ACTIONS):
        count = pred_hist["action"][action]
        lines.append(f"| {action} | {count:,} | {_pct(count, pred_total):.2f}% |")

    lines.extend([
        "",
        "Predicted technique counts (a row may carry several codes):",
        "",
        "| Technique | Count | % of rows |",
        "|---|---:|---:|",
    ])
    for code, count in pred_hist["techniques"].items():
        lines.append(f"| {code} | {count:,} | {_pct(count, pred_total):.2f}% |")

    lines.extend([
        "",
        "## 3. Predicted-0 cue rates vs predicted 1-4 (H2/H3)",
        "",
        "| Signal | All rows | Predicted 0 | Predicted 1-4 |",
        "|---|---:|---:|---:|",
    ])
    for col in CUE_COLUMNS:
        lines.append(
            f"| {col} | {_fmt_count_pct(cue['all'][col])} | "
            f"{_fmt_count_pct(cue['pred_0'][col])} | {_fmt_count_pct(cue['pred_1_4'][col])} |"
        )

    lines.extend([
        "",
        "## 4. Confidence by predicted class (H5)",
        "",
        "| Predicted class | n | Mean | q10 | q50 | q90 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key in ["all"] + [str(k) for k in range(config.NUM_CLASSES)]:
        if key in conf:
            item = conf[key]
            lines.append(
                f"| {key} | {item['n']:,} | {item['mean']:.4f} | {item['q10']:.4f} | "
                f"{item['q50']:.4f} | {item['q90']:.4f} |"
            )

    lines.extend([
        "",
        "## 5. Language mix (H4)",
        "",
        "| Bucket | Sample count | Sample % | Test count | Test % |",
        "|---|---:|---:|---:|---:|",
    ])
    sample_total_lang = sum(item["count"] for item in lang["sample"].values())
    test_total_lang = sum(item["count"] for item in lang["test"].values())
    for bucket in LANGUAGE_ORDER:
        s = lang["sample"][bucket]
        t = lang["test"][bucket]
        lines.append(
            f"| {bucket} | {s['count']:,} | {_pct(s['count'], sample_total_lang):.2f}% | "
            f"{t['count']:,} | {_pct(t['count'], test_total_lang):.2f}% |"
        )

    lines.extend([
        "",
        "Predicted class mix per test language bucket (counts only):",
        "",
        "| Bucket | n | " + " | ".join(str(k) for k in range(config.NUM_CLASSES)) + " |",
        "|---|---:|" + "---:|" * config.NUM_CLASSES,
    ])
    for bucket in LANGUAGE_ORDER:
        item = lang["pred_by_bucket"][bucket]
        counts = " | ".join(f"{item['label_counts'][str(k)]:,}" for k in range(config.NUM_CLASSES))
        lines.append(f"| {bucket} | {item['n']:,} | {counts} |")

    lines.extend(["", "## 6. M1 vs M2 argmax disagreement", ""])
    if disagree.get("available"):
        lines.append(
            f"On {disagree['n']:,} non-padded rows: {disagree['disagreements']:,} disagreements "
            f"({disagree['pct']:.2f}%)."
        )
    else:
        lines.append(f"Cannot compute: {disagree['reason']}.")

    p0_noise = cue["pred_0"]["noise_only"]["pct"]
    all_noise = cue["all"]["noise_only"]["pct"]
    cue_keys = ["has_url", "otp_like", "money", "cred", "urgency", "reward"]
    cue_excess = [k for k in cue_keys if cue["pred_0"][k]["pct"] > cue["pred_1_4"][k]["pct"]]
    cue_deficit = [k for k in cue_keys if cue["pred_0"][k]["pct"] < cue["pred_1_4"][k]["pct"]]
    lang_shift = max(
        abs(lang["test"][b]["pct"] - lang["sample"][b]["pct"]) for b in LANGUAGE_ORDER
    )
    conf0 = conf.get("0")
    conf14 = [conf[str(k)] for k in range(1, 5) if str(k) in conf]
    h5_low = bool(conf0 and conf14 and conf0["mean"] < conf["all"]["mean"]
                  and conf0["q10"] <= min(c["q10"] for c in conf14))

    lines.extend([
        "",
        "## Interpretation (hypotheses only; no test labels exist)",
        "",
        f"- **H1 M2 probabilities mostly padded:** "
        f"{'supported' if coverage['padded_pct'] >= 5.0 and coverage.get('ood_constant_tail', False) else 'not supported'}. "
        f"{coverage['padded_pct']:.2f}% of rows match the tail vector.",
        f"- **H2 Many predicted-0 rows are noise-only:** "
        f"{'supported' if p0_noise >= max(1.0, 2.0 * all_noise) else 'not supported'}. "
        f"Noise-only share is {p0_noise:.2f}% among predicted 0 vs {all_noise:.2f}% overall.",
        "- **H3 Predicted-0 rows contain scam cues:** cannot tell. "
        f"Higher for predicted 0: {', '.join(cue_excess) if cue_excess else 'none'}; "
        f"lower for predicted 0: {', '.join(cue_deficit) if cue_deficit else 'none'}. "
        "Cue presence is not a test label.",
        f"- **H4 Language shift hurts the model:** "
        f"{'supported' if lang_shift >= 5.0 else 'not supported'} for the shift itself "
        f"(largest bucket difference {lang_shift:.2f} points); effect on errors cannot tell without labels.",
        f"- **H5 Low confidence concentrates in class 0:** "
        f"{'supported' if h5_low else 'not supported'}. "
        f"Predicted-0 mean confidence is "
        f"{conf0['mean']:.4f} vs {conf['all']['mean']:.4f} overall." if conf0 else
        "- **H5 Low confidence concentrates in class 0:** cannot tell; predicted class 0 absent.",
        "",
    ])
    return "\n".join(lines)


def diagnose(submission_path: Path, test_path: Path, m2_path: Path,
             m1_path: Optional[Path] = None) -> Dict:
    """Load inputs and return aggregate-only diagnostic dictionaries."""
    sub = pd.read_csv(submission_path, dtype={"id": str}, keep_default_na=False)
    expected = ["id", "label", "techniques", "action", "confidence"]
    if list(sub.columns) != expected:
        raise ValueError(f"unexpected submission columns: {list(sub.columns)}")

    test = load_test()
    test_ids = test["id"].astype(str).to_numpy()
    if len(sub) != len(test) or not np.array_equal(sub["id"].to_numpy(), test_ids):
        raise ValueError("submission and test row order mismatch")
    prep = load_or_build_preprocessed_test(test).reset_index(drop=True)
    if (len(prep) != len(test)
            or not np.array_equal(prep["id"].astype(str).to_numpy(), test_ids)):
        raise ValueError("preprocessed cache and test row order mismatch")

    m2 = np.load(m2_path, allow_pickle=True)
    a2 = np.asarray(m2["A"], dtype=np.float64)
    if (len(a2) != len(test)
            or not np.array_equal(np.asarray(m2["ids"]).astype(str), test_ids)):
        raise ValueError("M2 arrays and test row order mismatch")

    m1_a = None
    if m1_path is not None and Path(m1_path).exists():
        with np.load(m1_path, allow_pickle=True) as m1:
            m1_a = np.asarray(m1["A"], dtype=np.float64)
        if (len(m1_a) != len(test)
                or not np.array_equal(np.asarray(m1["ids"]).astype(str), test_ids)):
            raise ValueError("M1 arrays and test row order mismatch")

    labels = sub["label"].to_numpy(dtype=np.int64)
    coverage = m2_coverage(a2, m2["ood_sim"])
    padded_mask = np.all(a2 == a2[-1], axis=1)
    return {
        "coverage": coverage,
        "sample_hist": histograms(
            load_sample()["label"], load_sample()["action"], load_sample()["techniques"]
        ),
        "pred_hist": histograms(labels, sub["action"], sub["techniques"]),
        "cue": cue_rates(prep, labels),
        "conf": confidence_stats(labels, sub["confidence"]),
        "lang": language_mix(load_sample()["text"], test["text"], labels),
        "disagree": m1_m2_disagreement(m1_a, a2, padded_mask),
    }


def main() -> str:
    """Write the aggregate diagnosis report and return its path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path,
                        default=config.PREDICTION_DIR / "submission_E1126_2221.csv")
    parser.add_argument("--test", type=Path, default=config.TEST_PATH)
    parser.add_argument("--m2", type=Path, default=config.PREDICTION_DIR / "E020_m2_test.npz")
    parser.add_argument("--m1-probs", type=Path, default=None)
    parser.add_argument("--output", type=Path,
                        default=config.VALIDATION_DIR / "test_diagnosis.md")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    diagnostics = diagnose(args.submission, args.test, args.m2, args.m1_probs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(diagnostics), encoding="utf-8")
    log.info("Diagnosis written to %s", args.output)
    return str(args.output)


if __name__ == "__main__":
    print(main())
