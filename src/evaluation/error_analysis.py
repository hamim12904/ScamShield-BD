"""Error-analysis artifacts for grouped-CV out-of-fold predictions."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src import config
from src.data_loader import LABEL_NAMES, techniques_to_multihot
from src.training.cross_validation import compute_slice_metrics

LABELS = list(range(6))


def _md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(y_true, y_pred, probs, df, name="analysis", tech_pred=None, output_dir=None):
    """Write confusion PNG and tabular error report for one OOF run."""
    out_dir = Path(output_dir) if output_dir is not None else Path(config.VALIDATION_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"{i} {LABEL_NAMES[i]}" for i in LABELS],
        yticklabels=[f"{i} {LABEL_NAMES[i]}" for i in LABELS],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} — confusion matrix (OOF)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    per_class = pd.DataFrame({
        "label": LABELS,
        "class": [LABEL_NAMES[i] for i in LABELS],
        "precision": p,
        "recall": r,
        "f1": f,
        "support": s,
    })
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    macro_row = pd.DataFrame([{
        "label": "macro",
        "class": "",
        "precision": macro_p,
        "recall": macro_r,
        "f1": macro_f,
        "support": int(s.sum()),
    }])
    per_class = pd.concat([per_class, macro_row], ignore_index=True)
    per_class.to_csv(out_dir / f"{name}_per_class.csv", index=False)

    slices = compute_slice_metrics(df, {"label": y_pred})
    slice_df = pd.DataFrame(sorted(slices.items()), columns=["slice", "macro_f1"])
    slice_df.to_csv(out_dir / f"{name}_slices.csv", index=False)

    errors = pd.DataFrame({
        "id": df["id"].to_numpy(),
        "text": df["text"].to_numpy(),
        "true": y_true,
        "pred": y_pred,
        "confidence": conf,
    })
    errors.to_csv(out_dir / f"{name}_errors.csv", index=False)
    legit_to_scam = errors[(errors["true"] == 0) & (errors["pred"] != 0)]
    scam_to_legit = errors[(errors["true"] != 0) & (errors["pred"] == 0)]

    if tech_pred is None:
        if "tech_pred" not in df.columns:
            raise ValueError("tech_pred is required for per-technique precision/recall")
        tech_pred = df["tech_pred"].to_numpy()
    tech_pred = np.asarray(tech_pred, dtype=np.int64)
    y_b = techniques_to_multihot(df["techniques"]).astype(np.int64)
    tech_rows = []
    for j, tech in enumerate(config.TECHNIQUES):
        tp, tr, tf, ts = precision_recall_fscore_support(
            y_b[:, j], tech_pred[:, j], labels=[0, 1], zero_division=0)
        tech_rows.append({
            "technique": tech,
            "precision": tp[1],
            "recall": tr[1],
            "f1": tf[1],
            "support": int(ts[1]),
            "predicted": int(tech_pred[:, j].sum()),
        })
    tech_df = pd.DataFrame(tech_rows)
    tech_df.loc[len(tech_df)] = {
        "technique": "macro",
        "precision": float(tech_df["precision"].mean()),
        "recall": float(tech_df["recall"].mean()),
        "f1": float(tech_df["f1"].mean()),
        "support": int(tech_df["support"].sum()),
        "predicted": int(tech_df["predicted"].sum()),
    }
    tech_df.to_csv(out_dir / f"{name}_techniques.csv", index=False)

    lines = [
        f"# Error report: {name}",
        "",
        "## Confusion matrix (true rows, predicted columns)",
        "",
        "| true \\ pred | " + " | ".join(str(i) for i in LABELS) + " |",
        "|---|" + "---|" * len(LABELS),
    ]
    for i in LABELS:
        lines.append(f"| {i} {LABEL_NAMES[i]} | " + " | ".join(str(v) for v in cm[i]) + " |")
    lines += ["", "![confusion](%s)" % f"{name}_confusion.png", "", "## Per-class metrics", "",
              _md_table(per_class), "", "## Slices (Task A macro-F1)", "",
              _md_table(slice_df), "", "## Per-technique metrics (thresholded predictions)", "",
              _md_table(tech_df), ""]

    lines += ["## Legitimate predicted as scam", ""]
    if legit_to_scam.empty:
        lines.append("None.")
    else:
        lines.append("| id | true | pred | confidence | text |")
        lines.append("|---|---|---|---|---|")
        for _, row in legit_to_scam.sort_values("confidence", ascending=False).iterrows():
            text = str(row["text"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['id']} | {row['true']} | {row['pred']} | {row['confidence']:.3f} | {text} |")
    lines += ["", "## Scam predicted as legitimate", ""]
    if scam_to_legit.empty:
        lines.append("None.")
    else:
        lines.append("| id | true | confidence | text |")
        lines.append("|---|---|---|---|")
        for _, row in scam_to_legit.sort_values("confidence", ascending=False).iterrows():
            text = str(row["text"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['id']} | {row['true']} | {row['confidence']:.3f} | {text} |")
    lines.append("")

    path = out_dir / f"{name}_error_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _run():
    from src.data_loader import load_sample

    z = np.load(config.OOF_DIR / "e001_m1.npz", allow_pickle=True)
    df = load_sample()
    order = pd.Series(np.arange(len(df)), index=df["id"])
    pos = order.loc[z["ids"]].to_numpy()
    df = df.iloc[pos].reset_index(drop=True)
    y_pred = z["A"].argmax(axis=1)
    tech_pred = (z["B"] >= 0.5).astype(np.int64)
    path = report(z["y_a"], y_pred, z["A"], df, "e001_m1", tech_pred=tech_pred)
    print(f"Wrote {path}")


if __name__ == "__main__":
    _run()
