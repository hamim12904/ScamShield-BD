import numpy as np
import pandas as pd

from src import config

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


def _read(path):
    # keep_default_na=False: texts like "NA", "null" or "nan" (possible noise-only
    # rows) must stay strings, and the techniques value "NONE" must not become NaN.
    df = pd.read_csv(path, keep_default_na=False, dtype={"text": str})
    df["text"] = df["text"].astype(str)
    return df


def load_sample():
    df = _read(config.SAMPLE_PATH)
    expected = ["id", "text", "label", "techniques", "action"]
    assert list(df.columns) == expected, f"sample.csv columns: {list(df.columns)}"
    df["label"] = df["label"].astype(int)
    df["action"] = df["action"].astype(int)
    return df


def load_test():
    df = _read(config.TEST_PATH)
    assert list(df.columns) == ["id", "text"], f"test.csv columns: {list(df.columns)}"
    return df


def load_submission_template():
    df = pd.read_csv(config.SAMPLE_SUBMISSION_PATH, keep_default_na=False)
    assert list(df.columns) == ["id", "label", "techniques", "action", "confidence"]
    return df


def parse_techniques(value):
    """'T1|T2' -> ['T1', 'T2'];  'NONE' -> []"""
    value = str(value).strip()
    if value == "" or value.upper() == "NONE":
        return []
    return [t.strip() for t in value.split("|") if t.strip()]


def techniques_to_multihot(series):
    """Series of 'T1|T2' strings -> array of shape (n, 6)."""
    out = np.zeros((len(series), len(config.TECHNIQUES)), dtype=int)
    for i, value in enumerate(series):
        for t in parse_techniques(value):
            out[i, config.TECHNIQUES.index(t)] = 1
    return out


def multihot_to_techniques(row):
    """array of 6 zeros/ones -> 'T1|T2' or 'NONE'"""
    codes = [t for t, on in zip(config.TECHNIQUES, row) if on]
    return "|".join(codes) if codes else "NONE"
