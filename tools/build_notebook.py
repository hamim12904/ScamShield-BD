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
    "src/training/train_llm_aug.py",
    "src/llm_label.py",
    "src/pseudo_label.py",
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

FROZEN_DATA_PATHS: tuple[str, ...] = (
    "data/extra/llm_labels_v1.jsonl",
    "data/extra/pseudo_labels_v1.jsonl",
    "outputs/validation_results/folds.csv",
)

DEFAULT_OUTPUT = Path("notebooks/notebook.ipynb")
DEFAULT_MEMORY = Path("MEMORY.md")


def _dotted_to_relpath(dotted: str) -> str | None:
    """Map a project dotted module name to its repository-relative file path."""
    parts = dotted.split(".")
    if parts[0] not in {"src", "tools", "data"}:
        return None
    return "/".join(parts) + ".py"


def validate_modules(module_paths: Sequence[str], repo_root: Path) -> None:
    """Fail the build on stub bodies or imports missing from the embedded module list."""
    embedded = set(module_paths)
    for rel_path in module_paths:
        full_path = repo_root / rel_path
        if not full_path.is_file():
            continue
        text = full_path.read_text(encoding="utf-8")
        if "NotImplementedError" in text:
            raise ValueError(f"Embedded module contains NotImplementedError: {rel_path}")

    for rel_path in module_paths:
        full_path = repo_root / rel_path
        if not full_path.is_file():
            continue
        text = full_path.read_text(encoding="utf-8")
        for match in re.finditer(r"^\s*from\s+([A-Za-z0-9_.]+)\s+import\s+([^\n#]+)", text, re.M):
            base, names_str = match.group(1), match.group(2)
            if base.split(".")[0] not in {"src", "tools", "data"}:
                continue
            folder = base.replace(".", "/")
            is_package = any(other.startswith(folder + "/") for other in embedded)
            if is_package:
                names = [n.strip().split(" as ")[0] for n in names_str.split(",")]
                candidates = [f"{base}.{name}" for name in names if name and name != "*"]
            else:
                candidates = [base]
            for dotted in candidates:
                rel = _dotted_to_relpath(dotted)
                if rel is None:
                    continue
                if rel not in embedded and (repo_root / rel).is_file():
                    raise ValueError(
                        f"Module {rel_path} imports {dotted} ({rel}) but it is not in MODULE_PATHS."
                    )
        for match in re.finditer(r"^\s*import\s+([A-Za-z0-9_.]+)", text, re.M):
            dotted = match.group(1)
            rel = _dotted_to_relpath(dotted)
            if rel is not None and rel not in embedded and (repo_root / rel).is_file():
                raise ValueError(
                    f"Module {rel_path} imports {dotted} ({rel}) but it is not in MODULE_PATHS."
                )


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
    return f"""# ScamShield BD — Reproducible Final Pipeline

> **Generated file:** Built by `tools/build_notebook.py`. Do not hand-edit this notebook.

## 1. Libraries & Environment
- Python 3.10–3.12; PyPI packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `ftfy`, `regex`, `unidecode`, `sentence-transformers`, `torch`, `python-dotenv` (exact runtime versions printed in Cell 2).
- Build-time dependency: `nbformat`. Clean-run packages are installed/verified in Cell 2.
- Licences: scientific Python libraries are BSD/MIT-compatible; the pretrained encoder is Apache-2.0 (below).

## 2. External Datasets, Pretrained Models & APIs (from `MEMORY.md` §11)
{section_11_text}

- **TF-IDF Vocabulary Policy:** In the final `prediction.main()` path, `fit_all()` passes no `df_test`, so TF-IDF word/char vocabulary and IDF are fit on **labeled + augmented training text only**. `TfidfBlock.fit()` can optionally include test text for unsupervised IDF when a caller supplies `test_d`, but that path is not used to produce the submitted candidate.
- **Model Features:** `prediction.py` uses word and char TF-IDF on `text_norm` (preserved text) and `text_skel` (skeleton), without message blocks or a raw-char branch.
- **Secrets / Keys:** Read via `os.getenv` / `python-dotenv`. `REGENERATE_SYNTH=False`, so the clean run uses frozen LLM/pseudo-label files and needs **no API key**. A key is required only when explicitly enabling regeneration.

## 3. Method & Validation Summary
- **Final model (Step 5 / E028):** 5x3 grouped CV with LR (E026) and LinearSVC (E027), promoted LLM + pseudo-label extras, full-test MiniLM Model 2 (E020), equal M1/M2 blend weights (0.50/0.50) for Tasks A/B/C, cross-fitted Task B thresholds, and a constant-confidence fallback. It writes `outputs/predictions/submission_E<mmdd>_<hhmm>.csv`, copied to `submission.csv`.
- **Validation:** Deterministic saved 5-fold × 3-repeat grouped CV (`outputs/validation_results/folds.csv`) with duplicate/template isolation and leave-cluster-out checks. Nested cross-fitted F_B=0.7103; calibrator Brier=0.1283 (constant baseline).
- **Corruption Robustness:** Set A augmentation is applied to training folds only (inheriting source `group_id`); Set B corruption is evaluation-only.

## 4. Reproducibility & Execution
- **Random Seed:** `PYTHONHASHSEED=42` and `config.set_seed(42)` for `random`, `numpy`, and `torch`; the embedded `folds.csv` is loaded, not rebuilt differently.
- **Run Order:** Install/version cell → setup → module and frozen-data `%%writefile` cells → imports/seeding → data location → Model 2 OOF + test embeddings (500,000 rows; GPU if available) → `prediction.main()` → strict validation → `submission.csv` → optional comparison with `SCAMSHIELD_COMPARE_SUBMISSION`.
- **Runtime:** Model 2 embeddings: about 6–9 min on a Kaggle T4/P100 GPU, roughly 3 h on CPU. The 500k-row preprocessing is cached and `predict_df` runs in 50,000-row chunks; emit a loud CPU warning before the full run. `SCAMSHIELD_TEST_SLICE=1000` executes a fast smoke test.
"""


def build_install_cell() -> str:
    """Return Cell 2 source: ensure required packages exist and print runtime versions."""
    return """import importlib
import importlib.metadata
import os
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
    ("regex", "regex"),
    ("ftfy", "ftfy"),
    ("unidecode", "Unidecode"),
    ("sentence_transformers", "sentence-transformers"),
    ("torch", "torch"),
    ("dotenv", "python-dotenv"),
]

for import_name, pip_name in REQUIRED_PACKAGES:
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"Installing missing package: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

print(f"Python:       {sys.version.split()[0]} ({platform.platform()})")
for pip_name in (
    "numpy", "pandas", "scipy", "scikit-learn", "joblib", "regex", "ftfy",
    "Unidecode", "sentence-transformers", "torch", "python-dotenv",
):
    try:
        print(f"{pip_name + ':':<22}{importlib.metadata.version(pip_name)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pip_name + ':':<22}unknown")
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
    if "NotImplementedError" in content:
        raise ValueError(f"Refusing to embed stub module containing NotImplementedError: {rel_path}")
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
from src.training import cross_validation, train_embeddings
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
    """Return run cell generating Model 2 artifacts then calling the final prediction pipeline."""
    return """t0 = time.perf_counter()
tracemalloc.start()

M2_EXP = "E020"
M2_OOF_PATH = config.OOF_DIR / f"{M2_EXP}_m2.npz"
M2_TEST_PATH = config.PREDICTION_DIR / f"{M2_EXP}_m2_test.npz"
expected_m2_rows = TEST_SLICE if TEST_SLICE is not None else len(df_test_full)

need_m2 = not M2_OOF_PATH.exists() or not M2_TEST_PATH.exists()
if not need_m2:
    embedded_ids = np.load(M2_TEST_PATH, allow_pickle=True)["ids"]
    need_m2 = len(embedded_ids) != expected_m2_rows

if need_m2:
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass
    if not cuda_available and TEST_SLICE is None:
        logging.warning(
            "CUDA unavailable: embedding 500,000 rows on CPU takes roughly 3 hours; "
            "use a Kaggle GPU for the full clean run."
        )
    logging.info("Generating Model 2 OOF and test probabilities (%s)...", M2_EXP)
    train_embeddings.run(
        exp_id=M2_EXP,
        test_limit=TEST_SLICE if TEST_SLICE is not None else None,
    )
    if TEST_SLICE is not None:
        m2_data = np.load(M2_TEST_PATH, allow_pickle=True)
        np.savez_compressed(
            M2_TEST_PATH,
            **{key: m2_data[key][:TEST_SLICE] for key in m2_data.files},
        )

assert M2_OOF_PATH.exists() and M2_TEST_PATH.exists(), "Model 2 artifacts were not generated"
assert len(np.load(M2_TEST_PATH, allow_pickle=True)["ids"]) == expected_m2_rows

print("Running final grouped CV, M1+M2 blend, calibration, final fit, and prediction...")
FINAL_SUBMISSION_PATH = prediction.main(test_limit=TEST_SLICE)
cv_elapsed = time.perf_counter() - t0
print(f"Final pipeline completed in {cv_elapsed:.2f}s")
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

root_sub = Path("submission.csv")
if root_sub.resolve() != SUBMISSION_PATH.resolve():
    sub_df.to_csv(root_sub, index=False, encoding="utf-8")

sample_for_check = SAMPLE_SUB_PATH
test_for_check = TEST_PATH
if TEST_SLICE is not None:
    sample_for_check = out_dir / "validation_results" / "sample_submission_smoke.csv"
    test_for_check = out_dir / "validation_results" / "test_smoke.csv"
    pd.read_csv(SAMPLE_SUB_PATH).head(TEST_SLICE).to_csv(sample_for_check, index=False)
    df_test_full.head(TEST_SLICE).to_csv(test_for_check, index=False)

errors = validate_submission.validate(
    SUBMISSION_PATH,
    sample_path=sample_for_check,
    test_path=test_for_check,
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
print(sub_df["label"].value_counts(normalize=True).sort_index().round(3))
print("\nConfidence summary:")
print(sub_df["confidence"].describe())
"""


def build_compare_cell() -> str:
    """Return final cell that compares the produced CSV with an optional reference path."""
    return r"""compare_env = os.getenv("SCAMSHIELD_COMPARE_SUBMISSION", "").strip()
if compare_env:
    reference_path = Path(compare_env)
    if not reference_path.is_file():
        raise FileNotFoundError(f"SCAMSHIELD_COMPARE_SUBMISSION not found: {reference_path}")
    reference_df = pd.read_csv(reference_path)
    print(f"DataFrame.equals vs {reference_path}: {sub_df.equals(reference_df)}")
    print(f"Byte equality vs {reference_path}: {SUBMISSION_PATH.read_bytes() == reference_path.read_bytes()}")
else:
    print("SCAMSHIELD_COMPARE_SUBMISSION not set; skipped CSV comparison.")
"""


def build_notebook(
    repo_root: Path = Path("."),
    output_path: Path = DEFAULT_OUTPUT,
    memory_path: Path = DEFAULT_MEMORY,
    *,
    allow_missing: bool = False,
) -> NotebookNode:
    """Assemble and write notebooks/notebook.ipynb from repo modules and MEMORY.md §11."""
    validate_modules(MODULE_PATHS, repo_root)
    section_11 = extract_memory_section_11(repo_root / memory_path)

    cells: list[NotebookNode] = [
        new_markdown_cell(build_declaration_markdown(section_11)),
        new_code_cell(build_install_cell()),
        new_code_cell(build_setup_cell(PACKAGE_DIRS)),
    ]

    for rel_path in MODULE_PATHS:
        cells.append(new_code_cell(build_writefile_cell(rel_path, repo_root, allow_missing)))
    for rel_path in FROZEN_DATA_PATHS:
        cells.append(new_code_cell(build_writefile_cell(rel_path, repo_root, allow_missing)))

    cells.extend(
        [
            new_code_cell(build_imports_cell()),
            new_code_cell(build_data_location_cell()),
            new_code_cell(build_cv_cell()),
            new_code_cell(build_predict_and_validate_cell()),
            new_code_cell(build_compare_cell()),
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
        "--ExecutePreprocessor.timeout=3600",
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
