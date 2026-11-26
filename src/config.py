from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
VALIDATION_DIR = OUTPUT_DIR / "validation_results"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

SAMPLE_PATH = DATA_DIR / "sample.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

SEED = 42
DELIMITER = "|||"

NUM_CLASSES = 6
TECHNIQUES = ["T1", "T2", "T3", "T4", "T5", "T6"]
NUM_ACTIONS = 6

for _d in (VALIDATION_DIR, PREDICTION_DIR):
    _d.mkdir(parents=True, exist_ok=True)