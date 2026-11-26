"""Fast unit tests for Stage 2 preprocessing & views contract."""
import unittest

from src.data_loader import load_sample
from src.preprocessing import (
    NOISE_VOCAB,
    build_views,
    is_noise_token,
    normalise_chars,
    preprocess,
)


class TestPreprocessing(unittest.TestCase):
    def test_idempotence(self):
        samples = [
            "S: Hі, check hxxps://daraz[.]com[.]bd/track ||| R: অরডার ৫০ টাকা",
            "mnbvcx Your bKash P1N will expire today!! 🎁",
            "Aarong: ne4r357 OUTLET +880 017XXXXXX73 TrxID 9K4L2M1X",
            "র\u200d্যা\u200bব অফার ২০% ছাড়",
        ]
        for text in samples:
            once = preprocess(text)["clean"]
            twice = preprocess(once)["clean"]
            self.assertEqual(once, twice)
            n_once = normalise_chars(text)
            self.assertEqual(n_once, normalise_chars(n_once))

    def test_preserves_delimiter_bangla_and_digits(self):
        conv = "S: আপনার পার্সেল ৫০ টাকা ||| R: ২০% ছাড় র\u200d্যাব"
        out = preprocess(conv)
        self.assertIn("|||", out["clean"])
        self.assertIn("50", out["clean"])
        self.assertIn("20%", out["clean"])
        self.assertIn("র\u200d্যা", out["clean"])
        self.assertNotIn("\u200b", out["clean"])
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

    def test_true_noise_only_rows(self):
        for bad in ("", "   ", None, float("nan"), "lkjsdfoiu xcvbnm", "zzqpwoeiru", "haswqzxgjsyh xkrdzrcrm fmbkjjea !!"):
            self.assertTrue(preprocess(bad)["noise_only"], f"{bad!r} should be noise_only")

    def test_noise_vocab_compliance(self):
        # Every token in NOISE_VOCAB must come from the problem statement or sample.csv
        sample_text = " ".join(load_sample()["text"].str.lower().tolist())
        for tok in NOISE_VOCAB:
            self.assertTrue(
                tok == "wkrjfejejrn" or tok in sample_text,
                f"Token {tok!r} not in problem statement or sample.csv",
            )
        self.assertFalse(is_noise_token("bkash"))
        self.assertTrue(is_noise_token("wkrjfejejrn"))

    def test_build_views_contract(self):
        df = load_sample().head(5)
        v = build_views(df)
        expected = {
            "text_raw", "text_norm", "text_skel", "n_msgs",
            "first_msg", "last_msg", "has_url", "n_urls",
            "noise_ratio", "script_bangla_ratio", "emoji_count",
        }
        self.assertTrue(expected.issubset(set(v.columns)))
        self.assertEqual(len(v), len(df))
        self.assertTrue((v["id"].values == df["id"].values).all())


if __name__ == "__main__":
    unittest.main()
