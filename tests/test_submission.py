"""Tests for tools/validate_submission.py (synthetic fixtures only; no real data needed)."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import validate_submission as vs

HEADER = ["id", "label", "techniques", "action", "confidence"]
IDS = [str(i) for i in range(20001, 20013)]  # 12 test ids


def good_rows():
    labels = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    techs = ["NONE", "T1|T2", "T4", "T6", "T1|T3", "NONE", "NONE", "T2", "T4", "T6", "T3", "NONE"]
    acts = [0, 1, 3, 4, 3, 0, 0, 1, 2, 4, 3, 0]
    confs = [0.91, 0.84, 0.77, 0.69, 0.88, 0.95, 0.9, 0.8, 0.7, 0.6, 0.85, 0.93]
    return [[i, str(l), t, str(a), str(c)] for i, l, t, a, c in zip(IDS, labels, techs, acts, confs)]


class TestValidator(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.sample = self.dir / "sample_submission.csv"
        self.test = self.dir / "test.csv"
        self._write(self.sample, HEADER, [[i, "0", "NONE", "0", "0.5"] for i in IDS])
        self._write(self.test, ["id", "text"], [[i, "hello"] for i in IDS])

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _write(path, header, rows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def run_check(self, rows, header=HEADER, raw=None):
        p = self.dir / "sub.csv"
        if raw is not None:
            p.write_bytes(raw)
        else:
            self._write(p, header, rows)
        return vs.check(p, sample=str(self.sample), test=str(self.test))

    def assertRejected(self, rows, fragment, **kw):
        errors, _ = self.run_check(rows, **kw)
        self.assertTrue(errors, "expected errors but file was accepted")
        self.assertTrue(any(fragment in e for e in errors), f"{fragment!r} not in {errors}")

    # ---- accepted ----
    def test_good_file_accepted(self):
        errors, warnings = self.run_check(good_rows())
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_validate_returns_list_and_cli_exit_codes(self):
        p = self.dir / "sub.csv"
        self._write(p, HEADER, good_rows())
        self.assertEqual(vs.validate(p, sample=str(self.sample), test=str(self.test)), [])
        argv = [str(p), "--sample", str(self.sample), "--test", str(self.test)]
        self.assertEqual(vs.main(argv), 0)
        rows = good_rows()
        rows[0][1] = "2.0"
        self._write(p, HEADER, rows)
        self.assertEqual(vs.main(argv), 1)

    # ---- rejected ----
    def test_float_label_rejected(self):
        rows = good_rows(); rows[0][1] = "2.0"
        self.assertRejected(rows, "'label' must be an integer")

    def test_label_out_of_range_and_text_rejected(self):
        for bad in ("6", "-1", "abc", " 1"):
            rows = good_rows(); rows[3][1] = bad
            self.assertRejected(rows, "'label' must be an integer")

    def test_action_float_rejected(self):
        rows = good_rows(); rows[2][3] = "3.0"
        self.assertRejected(rows, "'action' must be an integer")

    def test_empty_techniques_rejected(self):
        rows = good_rows(); rows[1][2] = ""
        self.assertRejected(rows, "empty cell(s) in column 'techniques'")

    def test_bad_techniques_rejected(self):
        for bad in ("T7", "T2|T1", "T1|T1", "NONE|T1", "t1", "T1,T2", "T1|"):
            rows = good_rows(); rows[1][2] = bad
            self.assertRejected(rows, "'techniques' must be NONE")

    def test_duplicate_id_rejected(self):
        rows = good_rows(); rows[5][0] = rows[4][0]
        self.assertRejected(rows, "duplicate ids")

    def test_missing_id_rejected(self):
        rows = good_rows()[:-1]
        errors, _ = self.run_check(rows)
        self.assertTrue(any("row count" in e for e in errors))
        self.assertTrue(any("missing from submission" in e for e in errors))

    def test_extra_id_rejected(self):
        rows = good_rows() + [["99999", "0", "NONE", "0", "0.5"]]
        errors, _ = self.run_check(rows)
        self.assertTrue(any("row count" in e for e in errors))
        self.assertTrue(any("not present in test.csv" in e for e in errors))

    def test_unknown_id_replacing_real_id_rejected(self):
        rows = good_rows(); rows[0][0] = "99999"
        errors, _ = self.run_check(rows)
        self.assertTrue(any("not present in test.csv" in e for e in errors))
        self.assertTrue(any("missing from submission" in e for e in errors))

    def test_confidence_out_of_range_rejected(self):
        for bad in ("1.2", "-0.1", "inf", "high"):
            rows = good_rows(); rows[0][4] = bad
            self.assertRejected(rows, "'confidence' must be a number in [0, 1]")

    def test_nan_confidence_rejected(self):
        for bad in ("nan", "NaN"):
            rows = good_rows(); rows[0][4] = bad
            self.assertRejected(rows, "'confidence' must be a number in [0, 1]")
        rows = good_rows(); rows[0][4] = ""
        self.assertRejected(rows, "empty cell(s) in column 'confidence'")

    def test_wrong_column_order_rejected(self):
        header = ["id", "label", "action", "techniques", "confidence"]
        rows = [[r[0], r[1], r[3], r[2], r[4]] for r in good_rows()]
        self.assertRejected(rows, "header", header=header)

    def test_two_column_and_extra_column_rejected(self):
        self.assertRejected([[r[0], r[1]] for r in good_rows()], "header", header=["id", "label"])
        self.assertRejected([r + ["x"] for r in good_rows()], "header", header=HEADER + ["text"])

    def test_index_column_rejected(self):
        rows = [[str(i)] + r for i, r in enumerate(good_rows())]
        errors, _ = self.run_check(rows, header=[""] + HEADER)
        self.assertTrue(any("index column" in e for e in errors))

    def test_bom_rejected(self):
        raw = vs.BOM + ("\n".join(",".join(r) for r in [HEADER] + good_rows()) + "\n").encode()
        self.assertRejected(None, "BOM", raw=raw)

    def test_non_utf8_rejected(self):
        self.assertRejected(None, "not valid UTF-8", raw=b"id,label\n\xff\xfe,1\n")

    def test_ragged_row_rejected(self):
        rows = good_rows(); rows[2] = rows[2][:-1]
        self.assertRejected(rows, "do not have 5 fields")

    def test_empty_cell_in_id_rejected(self):
        rows = good_rows(); rows[0][0] = ""
        self.assertRejected(rows, "empty cell(s) in column 'id'")

    # ---- warnings only ----
    def test_warnings_do_not_fail(self):
        rows = [[i, "1", "T1", "1", "0.9"] for i in IDS]
        errors, warnings = self.run_check(rows)
        self.assertEqual(errors, [])
        self.assertTrue(any("constant" in w for w in warnings))
        self.assertTrue(any("never predicted" in w for w in warnings))
        self.assertTrue(any("class 1" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()