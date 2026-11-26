"""Fast unit tests for Stage 2 preprocessing & 22-column views contract."""
import unittest

from src.data_loader import load_sample
from src.preprocessing import (
    NOISE_VOCAB,
    VIEW_CONTRACT_COLS,
    build_views,
    is_noise_token,
    normalise_chars,
    preprocess,
    to_skeleton,
)


class TestPreprocessing(unittest.TestCase):
    def test_idempotence(self):
        samples = [
            "S: Hі, check hxxps://daraz[.]com[.]bd/track ||| R: অরডার ৫০ টাকা",
            "mnbvcx Your bKash P1N will expire today!! 🎁",
            "Aarong: ne4r357 OUTLET +880 017XXXXXX73 TrxID 9K4L2M1X",
            "র\u200d্যা\u200bব অফার ২০% ছাড়",
            "Please v.e.r.i.f.y OTP 482913 now",
        ]
        for text in samples:
            p1 = preprocess(text)
            p2 = preprocess(p1["clean"])
            self.assertEqual(p1["clean"], p2["clean"])
            n_once = normalise_chars(text)
            self.assertEqual(n_once, normalise_chars(n_once))
            # Skeleton idempotence both directly and through preprocess()
            self.assertEqual(p1["skel"], to_skeleton(p1["skel"]))
            self.assertEqual(p1["skel"], preprocess(p1["skel"])["skel"])

    def test_cyrillic_and_leet_rules(self):
        self.assertEqual(preprocess("Vеrify")["clean"], "Verify")
        self.assertEqual(preprocess("0TP is 0")["clean"], "OTP is 0")

    def test_skeleton_spaced_letters_and_digit_shapes(self):
        self.assertEqual(preprocess("v.e.r.i.f.y")["skel"], "verify")
        self.assertEqual(preprocess("v e r i f y")["skel"], "verify")
        self.assertEqual(preprocess("OTP 482913")["skel"], "otp <N6>")
        self.assertEqual(to_skeleton("OTP 482913"), "otp <N6>")

    def test_preserves_delimiter_bangla_and_digits(self):
        conv = "S: আপনার পার্সেল ৫০ টাকা ||| R: ২০% ছাড় র\u200d্যাব"
        out = preprocess(conv)
        self.assertIn("|||", out["clean"])
        self.assertIn("50", out["clean"])
        self.assertIn("20%", out["clean"])
        self.assertIn("র\u200d্যা", out["clean"])
        self.assertNotIn("\u200b", out["clean"])
        self.assertEqual(out["n_messages"], 2)

    def test_n_messages_counts_raw_turns_even_if_noise(self):
        out = preprocess("Hello ||| zzqwpo")
        self.assertEqual(out["n_messages"], 2)

    def test_zero_width_and_nonempty(self):
        s = "He\u200bl\u200clo\u2060 world\ufeff"
        out = preprocess(s)
        self.assertEqual(out["clean"], "Hello world")
        self.assertTrue(len(out["clean"]) > 0)

    def test_short_real_words_not_noise_only(self):
        for w in ("Thanks", "Bkash", "Regards", "Hello", "Ok", "Urgent", "Parcel", "Lottery"):
            out = preprocess(w)
            self.assertFalse(out["noise_only"], f"{w!r} was falsely flagged noise_only")

    def test_consonant_cluster_words_not_dropped(self):
        for w in ("strengths", "twelfths", "month", "months"):
            self.assertFalse(is_noise_token(w), f"{w!r} was falsely flagged as noise")
            out = preprocess(w)
            self.assertFalse(out["noise_only"], f"{w!r} was falsely flagged noise_only")
            self.assertIn(w, out["clean"])
        with_noise = preprocess("zzqwpo strengths")
        self.assertFalse(with_noise["noise_only"])
        self.assertIn("strengths", with_noise["clean"])
        for w in ("wkrjfejejrn", "bcdfghjkl", "haswqzxgjsyh"):
            self.assertTrue(is_noise_token(w), f"{w!r} should be flagged as noise")

    def test_true_noise_only_rows(self):
        for bad in ("", "   ", None, float("nan"), "lkjsdfoiu xcvbnm", "zzqpwoeiru", "haswqzxgjsyh xkrdzrcrm fmbkjjea !!"):
            self.assertTrue(preprocess(bad)["noise_only"], f"{bad!r} should be noise_only")

    def test_noise_vocab_compliance(self):
        sample_text = " ".join(load_sample()["text"].str.lower().tolist())
        for tok in NOISE_VOCAB:
            self.assertTrue(
                tok == "wkrjfejejrn" or tok in sample_text,
                f"Token {tok!r} not in problem statement or sample.csv",
            )
        self.assertFalse(is_noise_token("bkash"))
        self.assertTrue(is_noise_token("wkrjfejejrn"))

    def test_build_views_22_column_contract(self):
        df = load_sample().head(8)
        v = build_views(df)
        self.assertEqual(len(VIEW_CONTRACT_COLS), 22)
        missing = set(VIEW_CONTRACT_COLS) - set(v.columns)
        self.assertEqual(missing, set(), f"Missing view columns: {missing}")
        self.assertEqual(len(v), len(df))
        self.assertTrue((v["id"].values == df["id"].values).all())
        # Verify OTP-like number flag on explicit 6-digit input
        otp_v = build_views(df.iloc[[0]].assign(text="Your OTP 482913 expires soon!"))
        self.assertTrue(bool(otp_v["has_otp_like_number"].iloc[0]))
        self.assertEqual(otp_v["text_skel"].iloc[0], "your otp <N6> expires soon!")


if __name__ == "__main__":
    unittest.main()