"""Word + char TF-IDF + structure feature block for Model 1.

Owned by Teammate B (T-010).
Vocabulary/IDF is fit on train text and, when `test_d` is supplied, test text (unsupervised; declared in notebook §11).
"""
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

from src import config
from src.preprocessing import preprocess

log = logging.getLogger(__name__)

# Token pattern that covers both ASCII and Bangla Unicode
BANGLA_TOKEN_PAT = r"(?u)[\w\u0980-\u09FF]+"


# ── Text-dict builder ─────────────────────────────────────────────────────────

def build_text_dict(df: pd.DataFrame) -> dict:
    """Build feature-ready text dict from a DataFrame with 'id' and 'text' columns.

    Always re-runs preprocessing on df['text'] so corrupted/modified text
    is handled correctly (critical for robustness evaluation in cross_validation).

    Returns dict with keys:
        text_norm, text_skel, text_raw, first_msg, last_msg, struct_feats
    """
    prep   = [preprocess(t) for t in df["text"]]
    norm   = [p["preserved"] for p in prep]
    skel   = [p["skel"] for p in prep]
    raw    = df["text"].fillna("").tolist()

    firsts, lasts = [], []
    for t in df["text"]:
        parts = str(t).split(config.DELIMITER)
        firsts.append(parts[0].strip())
        lasts.append(parts[-1].strip())

    struct = np.array(
        [
            [
                float(p["n_messages"]),
                float(p["n_noise_tokens"]),
                float(p["noise_only"]),
                float(p["had_lookalike"]),
                float(p["had_zero_width"]),
                float(p["had_emoji"]),
                len(str(r)) / 1000.0,          # raw_len normalised
            ]
            for p, r in zip(prep, raw)
        ],
        dtype=float,
    )

    return {
        "text_norm":    norm,
        "text_skel":    skel,
        "text_raw":     raw,
        "first_msg":    firsts,
        "last_msg":     lasts,
        "struct_feats": struct,
    }


# ── TfidfBlock ────────────────────────────────────────────────────────────────

class TfidfBlock:
    """Multi-view TF-IDF feature block.

    fit(train_d, test_d=None)  -> self
    transform(d)               -> csr_matrix
    fit_transform(train_d, ..) -> csr_matrix
    save(path) / load(path)    helpers
    """

    def __init__(self,
                 max_word_features: int = 200_000,
                 max_char_features: int = 200_000,
                 use_char_raw: bool = False,
                 use_msg_blocks: bool = True):
        self.max_word_features = max_word_features
        self.max_char_features = max_char_features
        self.use_char_raw = use_char_raw
        self.use_msg_blocks = use_msg_blocks

    @staticmethod
    def _coerce(seq) -> list:
        return [str(t) if t is not None else "" for t in (seq or [])]

    def fit(self, train_d: dict, test_d: Optional[dict] = None) -> "TfidfBlock":
        """Fit vectorizers. test_d texts are included for IDF only (unsupervised)."""

        def combine(key):
            vals = self._coerce(train_d.get(key, []))
            if test_d is not None:
                vals = vals + self._coerce(test_d.get(key, []))
            return vals

        norm  = combine("text_norm")
        skel  = combine("text_skel")
        first = combine("first_msg")
        last  = combine("last_msg")
        raw   = combine("text_raw")

        log.info("TfidfBlock: fitting on %d texts (train+test unsupervised)", len(norm))

        self.vw_ = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
            token_pattern=BANGLA_TOKEN_PAT, min_df=1,
            max_features=self.max_word_features,
        ).fit(norm)

        self.vc_ = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
            min_df=1, max_features=self.max_char_features,
        ).fit(skel)

        if self.use_char_raw:
            self.vcr_ = TfidfVectorizer(
                analyzer="char", ngram_range=(1, 4), sublinear_tf=True,
                min_df=1, max_features=50_000,
            ).fit(raw)

        if self.use_msg_blocks:
            self.vf_ = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                token_pattern=BANGLA_TOKEN_PAT, min_df=1, max_features=50_000,
            ).fit(first)
            self.vl_ = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                token_pattern=BANGLA_TOKEN_PAT, min_df=1, max_features=50_000,
            ).fit(last)

        struct_train = train_d.get("struct_feats")
        self.scaler_ = MaxAbsScaler().fit(struct_train) if struct_train is not None else None

        return self

    def transform(self, d: dict) -> csr_matrix:
        """Transform a texts dict to a stacked sparse feature matrix."""
        norm  = self._coerce(d.get("text_norm", []))
        skel  = self._coerce(d.get("text_skel", []))
        parts = [self.vw_.transform(norm), self.vc_.transform(skel)]

        if self.use_char_raw:
            raw = self._coerce(d.get("text_raw", []))
            parts.append(self.vcr_.transform(raw))

        if self.use_msg_blocks:
            first = self._coerce(d.get("first_msg", []))
            last  = self._coerce(d.get("last_msg",  []))
            parts.append(self.vf_.transform(first))
            parts.append(self.vl_.transform(last))

        struct = d.get("struct_feats")
        if struct is not None and self.scaler_ is not None:
            parts.append(csr_matrix(self.scaler_.transform(struct)))

        return hstack(parts).tocsr()

    def fit_transform(self, train_d: dict,
                      test_d: Optional[dict] = None) -> csr_matrix:
        """Convenience: fit on train_d (+ test_d for IDF), then transform train_d."""
        self.fit(train_d, test_d)
        return self.transform(train_d)

    def save(self, path) -> None:
        """Pickle the fitted block."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("TfidfBlock saved → %s", path)

    @classmethod
    def load(cls, path) -> "TfidfBlock":
        """Load a pickled TfidfBlock."""
        with open(path, "rb") as f:
            return pickle.load(f)
