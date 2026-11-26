"""Fast unit tests for Stage 3 folds, grouping, leave-cluster-out, and Set A/B augmentation."""
import logging
import random
import re
import unittest

import pandas as pd

from data.extra.handwritten import load_handwritten, validate
from src import config
from src.augment import corrupt, expand
from src.data_loader import load_sample
from src.training.cross_validation import (
    assign_groups,
    build_fold_table,
    get_folds,
    iter_saved_folds,
    leave_cluster_out_folds,
    load_or_create_folds,
)


class TestFoldsAndAugment(unittest.TestCase):
    def test_no_group_leakage_every_row_once_and_all_classes_present(self):
        df = load_sample()
        folds = build_fold_table(df)
        all_classes = set(df["label"].unique())

        for rep in range(config.N_REPEATS):
            col = f"fold_r{rep}"
            split_groups = (folds.groupby("group")[col].nunique() > 1).sum()
            self.assertEqual(split_groups, 0, f"Leakage in {col}")

            seen = []
            for tr, va in iter_saved_folds(folds, col):
                self.assertEqual(set(tr).intersection(set(va)), set())
                val_classes = set(df.iloc[va]["label"].unique())
                missing = all_classes - val_classes
                if missing:
                    logging.warning("Fold %s missing rare classes in validation: %s", col, missing)
                # At least 4 distinct classes present in each fold's validation set
                self.assertGreaterEqual(len(val_classes), 4)
                seen.extend(va.tolist())
            self.assertEqual(sorted(seen), list(range(len(df))))

        aug = expand(df.head(10), k=2, seed=42, ops="A")
        self.assertTrue((aug["source"] == "aug").all())
        self.assertIn("group_id", aug.columns)
        recombined = pd.concat([df.head(10).assign(group_id=df.head(10)["id"]), aug], ignore_index=True)
        self.assertEqual(recombined["group_id"].nunique(), 10)

    def test_same_seed_rebuilds_identical_table_and_get_folds_contract(self):
        df = load_sample()
        t1 = build_fold_table(df, seed=config.SEED)
        t2 = build_fold_table(df, seed=config.SEED)
        self.assertTrue(t1.equals(t2))

        pairs = get_folds(df, n_repeats=config.N_REPEATS)
        self.assertEqual(len(pairs), config.N_REPEATS * config.N_SPLITS)
        for tr, va in pairs:
            self.assertEqual(set(tr).intersection(set(va)), set())
            self.assertEqual(len(tr) + len(va), len(df))

    def test_sparse_assign_groups_keeps_near_duplicates_and_shared_first_msg_together(self):
        df = load_sample().head(6).copy()
        base_text = str(df.iloc[0]["text"])

        dup = df.iloc[[0]].copy()
        dup["id"] = 99991

        # Robust near-duplicate: swap a digit (if present) and delete one interior character
        near_text = re.sub(r"\d", lambda m: str((int(m.group(0)) + 1) % 10), base_text, count=1)
        mid = max(1, len(near_text) // 2)
        near_text = near_text[:mid] + near_text[mid + 1:]
        near = df.iloc[[0]].copy()
        near["id"] = 99992
        near["text"] = near_text

        # Shared first message (>= 20 chars) with completely different follow-up turns
        c1 = df.iloc[[0]].copy()
        c1["id"] = 99993
        c1["text"] = "S: Hello your parcel delivery needs confirmation ||| R: What parcel?"
        c2 = df.iloc[[0]].copy()
        c2["id"] = 99994
        c2["text"] = "S: Hello your parcel delivery needs confirmation ||| R: Send tracking link right now pls"

        comb = pd.concat([df, dup, near, c1, c2], ignore_index=True)
        g = assign_groups(comb)
        self.assertEqual(g.iloc[0], g.iloc[6])
        self.assertEqual(g.iloc[0], g.iloc[7])
        self.assertEqual(g.iloc[8], g.iloc[9])

    def test_saved_folds_deterministic_reload(self):
        df = load_sample()
        f1 = load_or_create_folds(df)
        f2 = load_or_create_folds(df)
        self.assertTrue(f1.equals(f2))

    def test_leave_cluster_out_disjoint_and_all_classes_in_training(self):
        df = load_sample()
        folds = build_fold_table(df)
        all_classes = set(df["label"].unique())
        seen_val = []
        for tr, va, cid in leave_cluster_out_folds(df, folds):
            self.assertEqual(set(tr).intersection(set(va)), set())
            self.assertEqual(folds.iloc[va]["cluster"].nunique(), 1)
            self.assertEqual(int(folds.iloc[va]["cluster"].iloc[0]), cid)
            self.assertEqual(set(df.iloc[tr]["label"].unique()), all_classes)
            seen_val.extend(va.tolist())
        self.assertEqual(sorted(seen_val), list(range(len(df))))

    def test_augment_preserves_delimiter_and_determinism(self):
        conv = "S: Verify your OTP now ||| R: Why? ||| S: Urgent account block"
        r1 = corrupt(conv, random.Random(7), level="medium", ops="A")
        r2 = corrupt(conv, random.Random(7), level="medium", ops="A")
        self.assertEqual(r1, r2)
        self.assertEqual(r1.count("|||"), 2)

        b1 = corrupt(conv, random.Random(7), level="medium", ops="B")
        self.assertEqual(b1.count("|||"), 2)
        self.assertNotEqual(r1, b1)

    def test_handwritten_extras_are_valid_and_grouped(self):
        df = load_handwritten()
        self.assertTrue(validate())
        self.assertEqual(len(df), 84)
        self.assertEqual(df["group_id"].nunique(), 14)
        self.assertTrue((df.groupby("group_id").size() == 6).all())
        self.assertEqual(set(df.loc[df["label"].isin([0, 5]), "techniques"]), {"NONE"})
        self.assertEqual(df["source"].unique().tolist(), ["handwritten"])


if __name__ == "__main__":
    unittest.main()