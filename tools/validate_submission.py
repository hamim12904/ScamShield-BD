"""Strict validator for the ScamShield BD 5-column Kaggle submission.

CLI:
    python tools/validate_submission.py <csv> [--sample data/sample_submission.csv] [--test data/test.csv]
Exit code 0 = valid, 1 = invalid (or unreadable).

Import:
    from tools.validate_submission import validate, check
    errors = validate("outputs/predictions/e001_1200.csv")          # list[str], empty = OK
    errors, warnings = check("outputs/predictions/e001_1200.csv")   # also returns warnings
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "data" / "sample_submission.csv"
DEFAULT_TEST = ROOT / "data" / "test.csv"

EXPECTED_COLS = ["id", "label", "techniques", "action", "confidence"]
VALID_TECH = {f"T{i}" for i in range(1, 7)}
BOM = b"\xef\xbb\xbf"
MAX_SHOWN = 5


def _show(ids: pd.Series) -> str:
    """Format up to MAX_SHOWN offending ids."""
    shown = ", ".join(map(str, ids.head(MAX_SHOWN).tolist()))
    more = f" (+{len(ids) - MAX_SHOWN} more)" if len(ids) > MAX_SHOWN else ""
    return f"[{shown}]{more}"


def _fail(errors: list, msg: str, ids: Optional[pd.Series] = None) -> None:
    if ids is not None:
        msg = f"{msg}: {len(ids):,} row(s), e.g. ids {_show(ids)}"
    errors.append(msg)


def _read_header(path: Path) -> Optional[list]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f), None)


def _read_ids(path: Path) -> pd.Series:
    df = pd.read_csv(path, usecols=["id"], dtype=str, keep_default_na=False)
    return df["id"]


def _load_submission(path: Path, errors: list) -> Optional[pd.DataFrame]:
    """Read the raw file as strings. Returns None if it cannot be parsed."""
    try:
        raw = path.read_bytes()
    except OSError as e:
        errors.append(f"cannot read file: {e}")
        return None

    if raw.startswith(BOM):
        errors.append("file starts with a UTF-8 BOM; save as plain UTF-8 without BOM")
        raw = raw[len(BOM):]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        errors.append(f"file is not valid UTF-8: {e}")
        return None

    rows = list(csv.reader(io.StringIO(text, newline="")))
    while rows and not rows[-1]:          # ignore trailing blank lines
        rows.pop()
    if not rows:
        errors.append("file is empty")
        return None

    header, body = rows[0], rows[1:]
    blank = [i + 2 for i, r in enumerate(body) if not r]
    if blank:
        errors.append(f"blank line(s) inside the file at line(s) {blank[:MAX_SHOWN]}")
        body = [r for r in body if r]

    ragged = [i + 2 for i, r in enumerate(body) if len(r) != len(header)]
    if ragged:
        errors.append(
            f"{len(ragged):,} row(s) do not have {len(header)} fields "
            f"(first at file line(s) {ragged[:MAX_SHOWN]})"
        )
        return None
    return pd.DataFrame(body, columns=header, dtype=str)


def check(
    path,
    sample: Optional[str] = None,
    test: Optional[str] = None,
) -> Tuple[list, list]:
    """Validate a submission. Returns (errors, warnings)."""
    errors: list = []
    warnings: list = []
    path = Path(path)

    df = _load_submission(path, errors)
    if df is None:
        return errors, warnings

    # 1. Header (and index-column detection)
    expected = EXPECTED_COLS
    sample_path = Path(sample) if sample else DEFAULT_SAMPLE
    if sample_path.exists():
        hdr = _read_header(sample_path)
        if hdr:
            expected = hdr
    elif sample:
        errors.append(f"sample submission not found: {sample_path}")
    else:
        warnings.append(f"sample_submission.csv not found at {sample_path}; using built-in header")

    cols = list(df.columns)
    if cols and cols[0] in ("", "Unnamed: 0", "index"):
        errors.append(f"first column {cols[0]!r} looks like a pandas index column; write with index=False")
    if cols != expected:
        errors.append(f"header {cols} != expected {expected} (names and order must match exactly)")
        return errors, warnings   # later checks need the right columns

    # 2. Row count and ids vs test.csv
    test_path = Path(test) if test else DEFAULT_TEST
    if test_path.exists():
        try:
            test_ids = _read_ids(test_path)
        except Exception as e:  # noqa: BLE001
            errors.append(f"cannot read test ids from {test_path}: {e}")
            test_ids = None
        if test_ids is not None:
            if len(df) != len(test_ids):
                errors.append(f"row count {len(df):,} != test.csv rows {len(test_ids):,}")
            ids = df["id"]
            dup = ids[ids.duplicated(keep=False)].drop_duplicates()
            if len(dup):
                _fail(errors, "duplicate ids", dup)
            test_set = set(test_ids)
            sub_set = set(ids)
            unknown = pd.Series(sorted(sub_set - test_set))
            missing = pd.Series(sorted(test_set - sub_set))
            if len(unknown):
                _fail(errors, "ids not present in test.csv", unknown)
            if len(missing):
                _fail(errors, "test ids missing from submission", missing)
    elif test:
        errors.append(f"test.csv not found: {test_path}")
    else:
        warnings.append(f"test.csv not found at {test_path}; skipped row-count and id checks")

    # 3. Empty cells (whitespace-only counts as empty)
    row_ids = df["id"]
    for col in df.columns:
        empty = df[col].str.strip() == ""
        if empty.any():
            _fail(errors, f"empty cell(s) in column '{col}'", row_ids[empty])
    nonempty = {c: df[c].str.strip() != "" for c in df.columns}

    # 4. label and action: integers 0-5 (reject 2.0, text, signs, spaces)
    for col in ("label", "action"):
        bad = nonempty[col] & ~df[col].str.fullmatch(r"[0-5]")
        if bad.any():
            _fail(errors, f"'{col}' must be an integer 0-5 (no decimals, text or spaces)", row_ids[bad])

    # 5. techniques
    def _tech_ok(v: str) -> bool:
        if v == "NONE":
            return True
        parts = v.split("|")
        return (
            all(p in VALID_TECH for p in parts)
            and len(set(parts)) == len(parts)
            and parts == sorted(parts)
        )

    tech = df["techniques"]
    bad = nonempty["techniques"] & ~tech.map(_tech_ok)
    if bad.any():
        _fail(
            errors,
            "'techniques' must be NONE or sorted unique T1-T6 joined by '|' (NONE never mixed with codes)",
            row_ids[bad],
        )

    # 6. confidence: numeric, finite, in [0, 1]
    conf = pd.to_numeric(df["confidence"], errors="coerce")
    bad = nonempty["confidence"] & ~((conf >= 0.0) & (conf <= 1.0))   # NaN/inf/text all fail
    if bad.any():
        _fail(errors, "'confidence' must be a number in [0, 1] (no NaN/inf/text)", row_ids[bad])

    # 7. Warnings only
    good_conf = conf[(conf >= 0.0) & (conf <= 1.0)]
    if len(good_conf) > 1 and good_conf.nunique() == 1:
        warnings.append(f"confidence is constant ({good_conf.iloc[0]}); calibration will not help")
    lab = df.loc[nonempty["label"] & df["label"].str.fullmatch(r"[0-5]"), "label"].astype(int)
    if len(lab):
        counts = lab.value_counts()
        never = [c for c in range(6) if c not in counts.index]
        if never:
            warnings.append(f"class(es) never predicted: {never}")
        top_share = counts.iloc[0] / len(lab)
        if top_share > 0.90:
            warnings.append(f"{top_share:.1%} of rows are predicted as class {counts.index[0]}")

    return errors, warnings


def validate(
    path,
    sample: Optional[str | Path] = None,
    test: Optional[str | Path] = None,
    sample_path: Optional[str | Path] = None,
    test_path: Optional[str | Path] = None,
) -> list:
    """Return a list of error strings; empty list means the file is valid."""
    s = sample if sample is not None else sample_path
    t = test if test is not None else test_path
    return check(path, sample=s, test=t)[0]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a ScamShield BD submission CSV.")
    ap.add_argument("csv", help="submission file to check")
    ap.add_argument("--sample", default=None, help=f"sample_submission.csv (default {DEFAULT_SAMPLE})")
    ap.add_argument("--test", default=None, help=f"test.csv (default {DEFAULT_TEST})")
    args = ap.parse_args(argv)

    errors, warnings = check(args.csv, sample=args.sample, test=args.test)
    for w in warnings:
        print(f"WARNING: {w}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        print(f"\nFAILED: {len(errors)} problem(s) in {args.csv}")
        return 1
    print(f"OK: {args.csv} passed all checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())