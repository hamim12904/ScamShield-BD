"""Validate a submission CSV against sample_submission.csv and test.csv.

NOTE: Owned by Teammate C (T-006). B created this stub to unblock T-011.
      C should replace/extend this with the full version. B will notify C.

CLI usage:
    python tools/validate_submission.py <csv> [--sample ...] [--test ...]
Exit 0 on all checks passed; exit 1 on any error. Warnings are non-fatal.

API usage:
    from tools.validate_submission import validate
    errors, warnings = validate(path, sample_path, test_path)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

VALID_TECHNIQUE_CODES = {"T1", "T2", "T3", "T4", "T5", "T6"}


def validate(path, sample_path=None, test_path=None):
    """Validate submission CSV. Returns (errors: list[str], warnings: list[str]).

    path:        path to submission CSV
    sample_path: path to sample_submission.csv (column-order check)
    test_path:   path to test.csv (row count + id check)
    """
    errors, warnings = [], []

    # --- Load -----------------------------------------------------------------
    try:
        sub = pd.read_csv(str(path), keep_default_na=False, dtype=str)
    except Exception as exc:
        return [f"Cannot read submission file: {exc}"], []

    # 1. Header / column order ------------------------------------------------
    expected_cols = ["id", "label", "techniques", "action", "confidence"]
    if sample_path and Path(sample_path).exists():
        tmpl = pd.read_csv(str(sample_path), keep_default_na=False, nrows=0)
        expected_cols = list(tmpl.columns)
    if list(sub.columns) != expected_cols:
        errors.append(
            f"Wrong column order. Expected {expected_cols}, got {list(sub.columns)}"
        )

    # 2. Row count vs test.csv ------------------------------------------------
    if test_path and Path(test_path).exists():
        test_df = pd.read_csv(str(test_path), keep_default_na=False,
                              dtype=str, usecols=["id"])
        if len(sub) != len(test_df):
            errors.append(
                f"Row count mismatch: submission {len(sub)}, test.csv {len(test_df)}"
            )
        sub_ids  = set(sub["id"].tolist())
        test_ids = set(test_df["id"].tolist())
        missing  = test_ids - sub_ids
        extra    = sub_ids - test_ids
        if missing:
            errors.append(f"Missing {len(missing)} test ids (e.g. {sorted(missing)[:5]})")
        if extra:
            errors.append(f"{len(extra)} unknown ids (e.g. {sorted(extra)[:5]})")

    # 3. Duplicate IDs --------------------------------------------------------
    if "id" in sub.columns:
        dups = sub["id"][sub["id"].duplicated()].tolist()
        if dups:
            errors.append(f"Duplicate ids ({len(dups)} rows, e.g. {dups[:5]})")

    # 4. label: integer string 0–5 -------------------------------------------
    if "label" in sub.columns:
        bad = []
        for i, v in enumerate(sub["label"]):
            try:
                iv = int(v)
                if str(iv) != str(v).strip() or not (0 <= iv <= 5):
                    bad.append((i, v))
            except (ValueError, TypeError):
                bad.append((i, v))
        if bad:
            errors.append(f"Invalid label in {len(bad)} rows (e.g. {bad[:5]})")

    # 5. action: integer string 0–5 ------------------------------------------
    if "action" in sub.columns:
        bad = []
        for i, v in enumerate(sub["action"]):
            try:
                iv = int(v)
                if str(iv) != str(v).strip() or not (0 <= iv <= 5):
                    bad.append((i, v))
            except (ValueError, TypeError):
                bad.append((i, v))
        if bad:
            errors.append(f"Invalid action in {len(bad)} rows (e.g. {bad[:5]})")

    # 6. techniques: NONE or sorted T1|T2 codes ------------------------------
    if "techniques" in sub.columns:
        bad = []
        for i, v in enumerate(sub["techniques"]):
            v = str(v).strip()
            if not v:
                bad.append((i, repr(v), "empty"))
                continue
            if v == "NONE":
                continue
            codes = v.split("|")
            if any(c not in VALID_TECHNIQUE_CODES for c in codes):
                bad.append((i, v, "invalid code"))
            elif len(codes) != len(set(codes)):
                bad.append((i, v, "duplicate codes"))
            elif codes != sorted(codes):
                bad.append((i, v, "not sorted"))
        if bad:
            errors.append(f"Invalid techniques in {len(bad)} rows (e.g. {bad[:5]})")

    # 7. confidence: float in [0, 1] -----------------------------------------
    if "confidence" in sub.columns:
        bad = []
        for i, v in enumerate(sub["confidence"]):
            try:
                fv = float(v)
                if not (0.0 <= fv <= 1.0):
                    bad.append((i, v))
            except (ValueError, TypeError):
                bad.append((i, v))
        if bad:
            errors.append(f"Invalid confidence in {len(bad)} rows (e.g. {bad[:5]})")

    # 8. No blank / NaN cells anywhere ---------------------------------------
    for col in sub.columns:
        blank = (sub[col].astype(str).str.strip() == "") | sub[col].isna()
        if blank.any():
            ids_eg = sub.loc[blank, "id"].tolist()[:5] if "id" in sub.columns else []
            errors.append(
                f"Blank/NaN in '{col}': {blank.sum()} rows"
                + (f" (e.g. ids {ids_eg})" if ids_eg else "")
            )

    # Soft warnings (non-fatal) -----------------------------------------------
    if "confidence" in sub.columns:
        try:
            if sub["confidence"].astype(float).nunique() == 1:
                warnings.append("All confidence values are identical")
        except Exception:
            pass

    if "label" in sub.columns:
        try:
            dist = sub["label"].value_counts(normalize=True)
            if dist.iloc[0] > 0.90:
                warnings.append(f"Over 90% of rows have label {dist.index[0]}")
            never = set(str(i) for i in range(6)) - set(sub["label"].tolist())
            if never:
                warnings.append(f"Classes never predicted: {sorted(never)}")
        except Exception:
            pass

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Validate submission CSV format"
    )
    parser.add_argument("csv", help="Submission CSV to validate")
    parser.add_argument("--sample", default="data/sample_submission.csv")
    parser.add_argument("--test",   default="data/test.csv")
    args = parser.parse_args()

    errors, warnings = validate(args.csv, args.sample, args.test)
    for w in warnings:
        print(f"[WARNING] {w}")
    if errors:
        for e in errors:
            print(f"[ERROR]   {e}")
        sys.exit(1)
    print(f"[OK] {args.csv} passed all checks.")
    sys.exit(0)


if __name__ == "__main__":
    main()
