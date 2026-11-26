"""Build the self-contained ScamShield BD Kaggle/Colab notebook via nbformat."""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import nbformat
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

logger = logging.getLogger(__name__)

# Dependency-ordered modules embedded via %%writefile cells
MODULE_PATHS: tuple[str, ...] = (
    "src/config.py",
    "src/data_loader.py",
    "data/extra/handwritten.py",
    "src/preprocessing.py",
    "src/augment.py",
    "src/text_analysis.py",
    "src/features/tfidf_features.py",
    "src/features/embedding_features.py",
    "src/models/task_a.py",
    "src/models/task_b.py",
    "src/models/task_c.py",
    "src/models/ensemble.py",
    "src/postprocess.py",
    "src/evaluation/calibration.py",
    "src/evaluation/metrics.py",
    "src/training/cross_validation.py",
    "src/training/train_baseline.py",
    "src/training/train_embeddings.py",
    "src/submission.py",
    "src/prediction.py",
    "tools/validate_submission.py",
)

PACKAGE_DIRS: tuple[str, ...] = (
    "data",
    "data/extra",
    "src",
    "src/evaluation",
    "src/features",
    "src/models",
    "src/training",
    "tools",
)

DEFAULT_OUTPUT = Path("notebooks/notebook.ipynb")
DEFAULT_MEMORY = Path("MEMORY.md")


def extract_memory_section_11(memory_path: Path) -> str:
    """Extract section 11 (external resources) from MEMORY.md; warn if missing or empty."""
    if not memory_path.is_file():
        logger.warning(
            "MEMORY.md not found at %s; inserting placeholder for section 11 declaration.",
            memory_path,
        )
        return (
            "No external pretrained models, external datasets, or external APIs are used. "
            "(WARNING: MEMORY.md was not found at build time — verify section 11 before release.)"
        )

    text = memory_path.read_text(encoding="utf-8")
    # Match Markdown heading for Section 11 (e.g., "## 11." or "## 11 " or "### 11.")
    pattern = re.compile(
        r"(?ms)^#{1,3}\s*11[\.\:\s][^\n]*\n(.*?)(?=^#{1,3}\s*\d+[\.\:\s]|\Z)"
    )
    match = pattern.search(text)
    if not match:
        logger.warning(
            "Section 11 heading not found in %s; inserting placeholder declaration.",
            memory_path,
        )
        return (
            "No external pretrained models, external datasets, or external APIs are used. "
            "(WARNING: Section 11 was missing in MEMORY.md — fill it before final packaging.)"
        )

    body = match.group(1).strip()
    cleaned = re.sub(r"<!--.*?-->", "", body, flags=re.S).strip()
    if not cleaned:
        logger.warning(
            "Section 11 in %s is empty; inserting default no-external-resources statement.",
            memory_path,
        )
        return (
            "No external pretrained models, external datasets, or external APIs are used. "
            "(WARNING: MEMORY.md section 11 was empty at build time — confirm and update MEMORY.md.)"
        )

    return cleaned


def build_declaration_markdown(section_11_text: str) -> str:
    """Assemble Cell 1 (required Markdown declaration) from MEMORY.md section 11."""
    return f"""# ScamShield BD — Reproducible Baseline Pipeline

> **Generated file:** Built by `tools/build_notebook.py`. Do not hand-edit this notebook.

## 1. Libraries & Environment
- Core Python stack: `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `python-dotenv` (exact runtime versions printed in Cell 2).
- Standard MIT/BSD-licensed scientific Python libraries only.

## 2. External Datasets, Pretrained Models & APIs (from `MEMORY.md` §11)
{section_11_text}

- **TF-IDF Vocabulary Policy:** All TF-IDF vectorizers are fit on **training data only** (never on test text).
- **Secrets / Keys:** Read exclusively via `os.getenv` / `python-dotenv`. No keys are hardcoded in this notebook.

## 3. Method & Validation Summary
- **Model:** E021 ensemble: multi-view TF-IDF (character skeleton + word + raw-char + first/last message + lexical cues) with class-balanced LR+LinearSVC M1 heads for Tasks A (`label`), B (`techniques`), and C (`action`), CV-tuned blend with declared Model 2 MiniLM E020 probabilities, a data-verified noise-only rule, and logistic confidence calibration with disagreement/OOD-similarity features.
- **Validation:** 5x3 grouped cross-validation (`folds.csv`) on real labeled rows only, preventing leakage across duplicate/template groups, with slice metrics and leave-cluster-out checks. The 84 declared hand-written minimal-pair rows keep their `group_id` and are adopted only if grouped CV improves.
- **Corruption Robustness:** Set A corruption augmentation is applied to training folds only (inheriting source `group_id`); Set B corruption is reserved for evaluation.

## 4. Reproducibility & Execution
- **Random Seed:** `SEED = 42` (`PYTHONHASHSEED=42`, numpy/random/sklearn derived from `config.SEED`).
- **Run Order:** Execute all cells top-to-bottom (`Run All`). The final cells train on labeled data, run `predict_df` on `test.csv`, validate the 5-column `submission.csv`, and verify compliance.
"""


def build_install_cell() -> str:
    """Return Cell 2 source: ensure required packages exist and print runtime versions."""
    return """import os
import platform
import subprocess
import sys

os.environ["PYTHONHASHSEED"] = "42"

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("dotenv", "python-dotenv"),
]

for import_name, pip_name in REQUIRED_PACKAGES:
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing missing package: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

print(f"Python:       {sys.version.split()[0]} ({platform.platform()})")
print(f"numpy:        {np.__version__}")
print(f"pandas:       {pd.__version__}")
print(f"scipy:        {scipy.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"joblib:       {joblib.__version__}")
"""


def build_setup_cell(package_dirs: Sequence[str]) -> str:
    """Return Cell 3 source: create package folders, __init__.py files, and load .env."""
    dirs_literal = repr(list(package_dirs))
    return f"""import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables without hardcoding any keys
load_dotenv()

for pkg_dir in {dirs_literal}:
    p = Path(pkg_dir)
    p.mkdir(parents=True, exist_ok=True)
    init_file = p / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")

for out_sub in (
    "outputs/validation_results",
    "outputs/oof",
    "outputs/predictions",
):
    Path(out_sub).mkdir(parents=True, exist_ok=True)

print("Package layout and environment initialized.")
"""


def build_writefile_cell(rel_path: str, repo_root: Path, allow_missing: bool) -> str:
    """Read a module from repo_root and wrap it in a %%writefile cell."""
    full_path = repo_root / rel_path
    if not full_path.is_file():
        if not allow_missing:
            raise FileNotFoundError(
                f"Required module not found: {full_path}. "
                "Re-run with --allow-missing only when building an early skeleton."
            )
        logger.warning("Missing module %s; emitting TODO stub cell (--allow-missing).", rel_path)
        return (
            f"%%writefile {rel_path}\n"
            f'"""TODO stub for {rel_path} (missing at skeleton build time)."""\n'
        )

    content = full_path.read_text(encoding="utf-8").rstrip() + "\n"
    return f"%%writefile {rel_path}\n{content}"


def build_imports_cell() -> str:
    """Return the imports and deterministic seeding cell placed after %%writefile cells."""
    return """import importlib
import logging
import os
import random
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

if "." not in sys.path:
    sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from src import config, data_loader, prediction, submission
from src.training import cross_validation
from tools import validate_submission

SEED = getattr(config, "SEED", 42)
if hasattr(config, "set_seed"):
    config.set_seed(SEED)
else:
    logging.warning("config.set_seed() not found; falling back to local random/numpy seeding.")
    random.seed(SEED)
    np.random.seed(SEED)

print(f"Modules imported successfully. SEED={SEED}, IS_KAGGLE={getattr(config, 'IS_KAGGLE', False)}")
"""


def build_data_location_cell() -> str:
    """Return Cell 4 source: locate dataset files via config.find_data_dir()."""
    return """if hasattr(config, "find_data_dir"):
    DATA_DIR = Path(config.find_data_dir())
else:
    logging.warning("config.find_data_dir() not found; falling back to Path('data').")
    DATA_DIR = Path("data")

SAMPLE_PATH = DATA_DIR / "sample.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

for p in (SAMPLE_PATH, TEST_PATH, SAMPLE_SUB_PATH):
    assert p.is_file(), f"Required data file missing: {p}"

TEST_SLICE_ENV = os.getenv("SCAMSHIELD_TEST_SLICE", "").strip()
TEST_SLICE = int(TEST_SLICE_ENV) if TEST_SLICE_ENV.isdigit() else None

if "path" in data_loader.load_sample.__code__.co_varnames:
    df_labeled = data_loader.load_sample(SAMPLE_PATH)
else:
    df_labeled = data_loader.load_sample()

df_test_full = pd.read_csv(TEST_PATH, dtype={"id": str})

if TEST_SLICE is not None and TEST_SLICE < len(df_test_full):
    logging.warning("SCAMSHIELD_TEST_SLICE=%d active — slicing test set for smoke test.", TEST_SLICE)
    df_test = df_test_full.head(TEST_SLICE).copy()
else:
    df_test = df_test_full

print(f"DATA_DIR:     {DATA_DIR}")
print(f"Labeled rows: {len(df_labeled)}")
print(f"Test rows:    {len(df_test)} (full={len(df_test_full)})")
"""


def build_cv_cell() -> str:
    """Return run cell for the full E021 CV/blend/calibration/final-fit/prediction pipeline."""
    return """t0 = time.perf_counter()
tracemalloc.start()

print("Running E021 grouped CV, M1+M2 blend, calibration, final fit, and prediction...")
FINAL_SUBMISSION_PATH = prediction.main()
cv_elapsed = time.perf_counter() - t0
print(f"E021 full pipeline completed in {cv_elapsed:.2f}s")
"""


def build_predict_and_validate_cell() -> str:
    """Return run cell for test inference via predict_df, writing submission.csv, and validation."""
    return r"""t1 = time.perf_counter()
sub_df = pd.read_csv(FINAL_SUBMISSION_PATH)
pred_elapsed = time.perf_counter() - t1

out_dir = Path(getattr(config, "OUT_DIR", Path(".")))
out_dir.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = out_dir / "submission.csv"
sub_df.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8")

# Also write to working directory root if OUT_DIR differs (convenient for Kaggle submission)
root_sub = Path("submission.csv")
if root_sub.resolve() != SUBMISSION_PATH.resolve():
    sub_df.to_csv(root_sub, index=False, encoding="utf-8")

errors = validate_submission.validate(
    SUBMISSION_PATH,
    sample_path=SAMPLE_SUB_PATH,
    test_path=TEST_PATH,
)
if errors:
    raise RuntimeError("Submission validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

_, peak_mem_bytes = tracemalloc.get_traced_memory()
tracemalloc.stop()
total_elapsed = time.perf_counter() - t0

print(f"VALIDATION PASSED: {SUBMISSION_PATH}")
print(
    f"Prediction load/validate: {pred_elapsed:.2f}s | Total wall time: {total_elapsed:.2f}s | "
    f"Peak traced RAM: {peak_mem_bytes / (1024 ** 2):.1f} MB"
)
print("\nClass distribution:")
print(sub_df["label"].value_counts().sort_index())
print("\nConfidence summary:")
print(sub_df["confidence"].describe())
"""


def build_notebook(
    repo_root: Path = Path("."),
    output_path: Path = DEFAULT_OUTPUT,
    memory_path: Path = DEFAULT_MEMORY,
    *,
    allow_missing: bool = False,
) -> NotebookNode:
    """Assemble and write notebooks/notebook.ipynb from repo modules and MEMORY.md §11."""
    section_11 = extract_memory_section_11(repo_root / memory_path)

    cells: list[NotebookNode] = [
        new_markdown_cell(build_declaration_markdown(section_11)),
        new_code_cell(build_install_cell()),
        new_code_cell(build_setup_cell(PACKAGE_DIRS)),
    ]

    for rel_path in MODULE_PATHS:
        cells.append(new_code_cell(build_writefile_cell(rel_path, repo_root, allow_missing)))

    cells.extend(
        [
            new_code_cell(build_imports_cell()),
            new_code_cell(build_data_location_cell()),
            new_code_cell(build_cv_cell()),
            new_code_cell(build_predict_and_validate_cell()),
        ]
    )

    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
        },
    )

    out_file = repo_root / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    logger.info("Wrote %d cells to %s", len(cells), out_file)
    return nb


def run_smoke_test(notebook_path: Path, test_slice: int = 1000) -> int:
    """Execute the generated notebook via jupyter nbconvert on a sliced test set."""
    import os

    env = os.environ.copy()
    env["SCAMSHIELD_TEST_SLICE"] = str(test_slice)
    executed_out = notebook_path.with_name(f"{notebook_path.stem}.executed.ipynb")
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook_path),
        "--output",
        executed_out.name,
        "--output-dir",
        str(notebook_path.parent),
    ]
    logger.info("Running smoke test: %s (SCAMSHIELD_TEST_SLICE=%d)", " ".join(cmd), test_slice)
    proc = subprocess.run(cmd, env=env, check=False)
    return proc.returncode


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for build_notebook.py."""
    parser = argparse.ArgumentParser(
        description="Build notebooks/notebook.ipynb from src/ modules and MEMORY.md section 11."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root directory (default: current directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output notebook path relative to repo root (default: notebooks/notebook.ipynb).",
    )
    parser.add_argument(
        "--memory",
        type=Path,
        default=DEFAULT_MEMORY,
        help="Path to MEMORY.md relative to repo root (default: MEMORY.md).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Emit TODO stub cells for missing modules (for early skeleton builds only).",
    )
    parser.add_argument(
        "--execute-slice",
        type=int,
        default=None,
        help="Optional: execute notebook via jupyter nbconvert on first N test rows (e.g., 1000).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for building (and optionally smoke-testing) the notebook."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    args = _build_parser().parse_args(argv)
    build_notebook(
        repo_root=args.repo_root,
        output_path=args.output,
        memory_path=args.memory,
        allow_missing=args.allow_missing,
    )
    if args.execute_slice is not None:
        return run_smoke_test(args.repo_root / args.output, test_slice=args.execute_slice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
