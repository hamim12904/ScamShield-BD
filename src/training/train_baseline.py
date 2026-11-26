"""Stage 4: baseline + first submission.   python -m src.training.train_baseline"""
import random

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import config
from src.data_loader import (load_sample, load_submission_template, load_test,
                             multihot_to_techniques, techniques_to_multihot)
from src.preprocessing import preprocess
from src.training.cross_validation import corrupt_text, cross_validate


class Baseline:
    def __init__(self, n_aug=3, seed=config.SEED):
        self.n_aug, self.seed = n_aug, seed

    def _vec(self, clean):
        return hstack([self.vc.transform(clean), self.vw.transform(clean)]).tocsr()

    def fit(self, train):
        rng = random.Random(self.seed)
        texts, y = list(train.text), list(train.label)
        for t, l in zip(train.text, train.label):          # corrupted copies (declare in notebook)
            for _ in range(self.n_aug):
                texts.append(corrupt_text(t, rng))
                y.append(l)
        clean = [preprocess(t)["clean"].lower() for t in texts]
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self.vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                                  token_pattern=r"(?u)\b\w+\b")
        self.vc.fit(clean)
        self.vw.fit(clean)
        self.clf = LogisticRegression(C=20, max_iter=3000, class_weight="balanced")
        self.clf.fit(self._vec(clean), y)

        # Tasks B and C: most common action / techniques for each class in the real rows
        self.action_by_label = train.groupby("label")["action"].agg(lambda s: s.mode().iloc[0]).to_dict()
        tech = techniques_to_multihot(train.techniques)
        self.tech_by_label = {}
        for l in sorted(train.label.unique()):
            m = tech[(train.label == l).values].mean(0)
            arr = (m >= 0.4).astype(int)
            if m.max() > 0 and arr.sum() == 0:
                arr[m.argmax()] = 1
            self.tech_by_label[l] = arr
        return self

    def predict(self, df):
        prep = [preprocess(t) for t in df["text"]]
        clean = [p["clean"].lower() for p in prep]
        proba = np.zeros((len(df), config.NUM_CLASSES))
        proba[:, self.clf.classes_] = self.clf.predict_proba(self._vec(clean))
        noise = np.array([p["noise_only"] for p in prep])
        proba[noise] = [0.60, 0.08, 0.08, 0.08, 0.08, 0.08]   # noise-only -> Legitimate
        label = proba.argmax(1)
        return {
            "label": label,
            "proba": proba,
            "action": np.array([self.action_by_label[l] for l in label]),
            "tech": np.array([self.tech_by_label[l] for l in label]),
        }


def main():
    df = load_sample()
    df["group"] = df["id"]
    res, summary = cross_validate(df, lambda tr, va: Baseline().fit(tr).predict(va), n_repeats=3)
    print(summary)
    res.to_csv(config.VALIDATION_DIR / "baseline_cv.csv", index=False)

    model = Baseline().fit(df)
    test, template = load_test(), load_submission_template()
    pred = model.predict(test)
    sub = pd.DataFrame({
        "id": test["id"],
        "label": pred["label"].astype(int),
        "techniques": [multihot_to_techniques(r) for r in pred["tech"]],
        "action": pred["action"].astype(int),
        "confidence": np.clip(pred["proba"].max(1), 0.30, 0.98).round(4),
    })
    assert (sub["id"].values == template["id"].values).all() and len(sub) == len(template)
    sub.to_csv(config.PREDICTION_DIR / "submission_full.csv", index=False)       # 5 columns
    sub[["id", "label"]].to_csv(config.PREDICTION_DIR / "submission.csv", index=False)  # 2 columns
    print("\npredicted class mix:\n", sub["label"].value_counts(normalize=True).sort_index().round(3))


if __name__ == "__main__":
    main()