"""Fast unit tests for Stage 3 folds, sparse grouping, leave-cluster-out, and Set A/B augmentation."""
import random
import unittest

import pandas as pd

from src.augment import corrupt, expand
from src.data_loader import load_sample
from src.training.cross_validation import (
    assign_groups,
    build_fold_table,
    iter_saved_folds,
    leave_cluster_out_folds,
    load_or_create_folds,
)


class TestFoldsAndAugment(unittest.TestCase):
    def test_no_group_leakage_across_repeats(self):
        df = load_sample()
        folds = build_fold_table(df)
        for rep in range(3):
            col = f"fold_r{rep}"
            split_groups = (folds.groupby("group")[col].nunique() > 1).sum()
            self.assertEqual(split_groups, 0, f"Leakage in {col}")

        # Augmented rows inherit source group and never cross folds
        aug = expand(df.head(10), k=2, seed=42, ops="A")
        self.assertTrue((aug["source"] == "aug").all())
        self.assertTrue("group_id" in aug.columns)
        recombined = pd.concat([df.head(10).assign(group_id=df.head(10)["id"]), aug], ignore_index=True)
        # Grouping by group_id keeps all variants with source
        self.assertEqual(recombined["group_id"].nunique(), 10)

    def test_sparse_assign_groups_keeps_near_duplicates_together(self):
        df = load_sample().head(6).copy()
        dup = df.iloc[[0]].copy()
        dup["id"] = 99991
        near = df.iloc[[0]].copy()
        near["id"] = 99992
        near["text"] = near["text"].iloc[0].replace("29 Nov", "30 Dec").replace("shipped", "shpped")
        comb = pd.concat([df, dup, near], ignore_index=True)
        g = assign_groups(comb)
        self.assertEqual(g.iloc[0], g.iloc[6])
        self.assertEqual(g.iloc[0], g.iloc[7])

    def test_saved_folds_deterministic_reload(self):
        df = load_sample()
        f1 = load_or_create_folds(df)
        f2 = load_or_create_folds(df)
        self.assertTrue(f1.equals(f2))
        for tr, va in iter_saved_folds(f1, "fold_r0"):
            self.assertEqual(set(tr).intersection(set(va)), set())
            self.assertEqual(len(tr) + len(va), len(df))

    def test_leave_cluster_out_disjoint(self):
        df = load_sample()
        folds = build_fold_table(df)
        seen_val = []
        for tr, va, cid in leave_cluster_out_folds(df, folds):
            self.assertEqual(set(tr).intersection(set(va)), set())
            self.assertEqual(folds.iloc[va]["cluster"].nunique(), 1)
            self.assertEqual(int(folds.iloc[va]["cluster"].iloc[0]), cid)
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


if __name__ == "__main__":
    unittest.main()
