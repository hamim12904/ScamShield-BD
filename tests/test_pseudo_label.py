"""Unit tests for confidence-gated pseudo-labeling (T-032)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_sample
from src.prediction import fit_all
from src.pseudo_label import (
    load_pseudo_labels,
    mine_pseudo_labels,
    save_pseudo_labels,
)
from src.training.train_llm_aug import compute_near_real_ids


class TestPseudoLabeling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real_df = load_sample()
        cls.fitted = fit_all(cls.real_df, n_aug=1)

    def test_pre_filtering_and_capping(self):
        # Create synthetic test rows including noise-only and short text
        mock_test = pd.DataFrame({
            "id": [30001, 30002, 30003, 30004, 30005],
            "text": [
                "short",  # < 12 chars -> should be dropped
                "wkrjfejejrn wkrjfejejrn wkrjfejejrn",  # noise-only -> should be dropped
                "bKash agent bolchi, apnaar account block hobe, OTP bolen.",  # fraud
                "Congratulations! You won 1,000,000 taka in lottery! Click here.",  # fraud
                "Your parcel delivery is arriving today. Thank you for shopping with us.",  # legit
            ]
        })

        mined = mine_pseudo_labels(
            mock_test,
            self.fitted,
            conf_threshold=0.30,  # low threshold for testing mock sample
            max_per_class=2,
        )

        self.assertNotIn(30001, mined["id"].values)
        self.assertNotIn(30002, mined["id"].values)
        self.assertTrue((mined["weight"] == 0.20).all())
        self.assertTrue((mined["source"] == "pseudo").all())

    def test_multi_task_rule_consistency(self):
        mock_test = pd.DataFrame({
            "id": [40001, 40002],
            "text": [
                "Your mobile bill payment of 500 BDT was successful. Transaction ID: TXN12345.",
                "bKash theke 25,000 taka jitechen! PIN share korun taka pete.",
            ]
        })

        mined = mine_pseudo_labels(
            mock_test,
            self.fitted,
            conf_threshold=0.30,
            max_per_class=10,
        )

        for _, row in mined.iterrows():
            if row["label"] in (0, 5):
                self.assertEqual(row["techniques"], "NONE")
                self.assertEqual(row["action"], 0)

    def test_save_and_load_roundtrip(self):
        mock_df = pd.DataFrame({
            "id": [50001, 50002],
            "text": ["Genuine transaction notification.", "Urgent verify your bank PIN now."],
            "label": [0, 1],
            "techniques": ["NONE", "T1,T5"],
            "action": [0, 1],
            "source": ["pseudo", "pseudo"],
            "weight": [0.20, 0.20],
            "confidence": [0.92, 0.95],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "test_pseudo.jsonl"
            save_pseudo_labels(mock_df, tmp_path)
            loaded = load_pseudo_labels(tmp_path)

            self.assertEqual(len(loaded), 2)
            self.assertEqual(list(loaded["id"]), [50001, 50002])
            self.assertEqual(list(loaded["label"]), [0, 1])
            self.assertEqual(list(loaded["techniques"]), ["NONE", "T1|T5"])
            self.assertEqual(list(loaded["action"]), [0, 1])
            self.assertTrue((loaded["source"] == "llm").all() or (loaded["source"] == "pseudo").all())

    def test_leak_free_near_real_detection(self):
        # Row 1 is exact duplicate of real row 10001
        real_text = self.real_df.iloc[0]["text"]
        mock_pseudo = pd.DataFrame({
            "id": [60001, 60002],
            "text": [real_text, "A totally unique and different string without similarity xyz123."],
            "label": [int(self.real_df.iloc[0]["label"]), 0],
            "techniques": ["NONE", "NONE"],
            "action": [0, 0],
            "source": ["pseudo", "pseudo"],
            "weight": [0.20, 0.20],
        })

        result = compute_near_real_ids(mock_pseudo, self.real_df)
        self.assertIn("near_real_ids", result.columns)
        # First row must match real row's id
        real_id = self.real_df.iloc[0]["id"]
        self.assertIn(real_id, result.iloc[0]["near_real_ids"])
        # Second row should have no matches
        self.assertEqual(len(result.iloc[1]["near_real_ids"]), 0)


if __name__ == "__main__":
    unittest.main()
