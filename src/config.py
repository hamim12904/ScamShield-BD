"""Shared configuration: seeds, paths, constants.

Owned by Teammate B (T-001). Everyone imports this; create first.
"""
import os
import random
from pathlib import Path

import numpy as np


# ── Root ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
IS_KAGGLE = os.path.exists("/kaggle")


# ── Helpers ───────────────────────────────────────────────────────────────────
def find_data_dir() -> Path:
    """Search for the data directory containing sample.csv (works locally and on Kaggle)."""
    candidates = [
        ROOT / "data",
        ROOT.parent / "data",
        Path("..") / "data",
        Path(".") / "data",
        Path("/kaggle/input"),
    ]
    for c in candidates:
        if (c / "sample.csv").exists():
            return c
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for sub in kaggle_input.iterdir():
            if sub.is_dir() and (sub / "sample.csv").exists():
                return sub
        nested = find_nested_data_dir(kaggle_input)
        if nested is not None:
            return nested
    return ROOT / "data"   # default; loader will raise a clear error if missing


def find_nested_data_dir(base: Path) -> Path | None:
    """Return the sorted-first directory under base that contains sample.csv."""
    if not base.exists():
        return None
    for path in sorted(base.rglob("sample.csv")):
        if path.is_file():
            return path.parent
    return None


def set_seed(seed: int = 42) -> None:
    """Seed random, numpy, and torch (if installed) for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── Run-time constants ────────────────────────────────────────────────────────
SEED = 42
DELIMITER = "|||"
N_SPLITS = 5
N_REPEATS = 3
MAX_LEN = 256
CONF_CLIP = (0.03, 0.97)
USE_LLM = bool(os.getenv("LLM_API_KEY"))
REGENERATE_SYNTH = False   # True = call API; False = load frozen file

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = find_data_dir()
OUT_DIR = Path("/kaggle/working") if IS_KAGGLE else ROOT / "outputs"
OUTPUT_DIR = ROOT / "outputs"
VALIDATION_DIR = OUTPUT_DIR / "validation_results"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
CACHE_DIR = OUTPUT_DIR / "cache"
OOF_DIR = OUTPUT_DIR / "oof"

SAMPLE_PATH = DATA_DIR / "sample.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

# ── Task definitions ──────────────────────────────────────────────────────────
NUM_CLASSES = 6
NUM_ACTIONS = 6
TECHNIQUES = ["T1", "T2", "T3", "T4", "T5", "T6"]

LABEL_NAMES = {
    0: "Legitimate",
    1: "MFS/OTP scam",
    2: "Prize/lottery scam",
    3: "Fake job/investment",
    4: "Delivery/parcel phishing",
    5: "Promotional spam",
}

ACTION_NAMES = {
    0: "No action",
    1: "Share credentials",
    2: "Pay / transfer money",
    3: "Click link",
    4: "Install / join / task",
    5: "Contact sender",
}

# ── Bootstrap output directories ──────────────────────────────────────────────
for _d in (VALIDATION_DIR, PREDICTION_DIR, CACHE_DIR, OOF_DIR):
    _d.mkdir(parents=True, exist_ok=True)
