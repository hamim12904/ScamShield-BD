"""Stage 2: normalisation, noise handling, conversation parsing.

Run from the project root:
    python -m src.preprocessing
"""
import re
import unicodedata

import ftfy
import pandas as pd

from src import config

# --- Unicode lookalikes (Cyrillic / Greek -> Latin) -------------------------
_LOOKALIKE_PAIRS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ԛ": "q", "ԝ": "w", "ı": "i",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S", "У": "Y",
    "ο": "o", "ν": "v", "ρ": "p", "ι": "i", "Α": "A", "Β": "B", "Ε": "E",
    "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Χ": "X",
}
LOOKALIKES = str.maketrans(_LOOKALIKE_PAIRS)

BANGLA = "\u0980-\u09FF"
ZW_ALWAYS_RE = re.compile("[\u200b\u2060\ufeff\u00ad]")
# ZWNJ/ZWJ are legitimate inside Bangla words, so only strip them elsewhere
ZWNJ_RE = re.compile("\u200c")
ZWJ_RE = re.compile("\u200d(?!্)|(?<!র)\u200d")   # keeps only র + ZWJ + ্
EMOJI_RE = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE00-\uFE0F]")
REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")          # letters only: keeps 5000 intact
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[a-z0-9-]+\.(?:com|net|org|bd|co|io|xyz|link|info|top|vip|me|ly)\b\S*",
    re.I,
)
SPEAKER_RE = re.compile(r"^\s*[\[\(]?([A-Za-z]{1,10})[\]\)]*\s*:\s*")
WORD_RE = re.compile(r"[A-Za-z]+")
NOISE_WHITELIST = {"strengths", "lengths", "twelfths"}


def is_noise_token(tok):
    """Heuristic for strings like 'wkrjfejejrn'. Review the flagged list on test!"""
    tok = tok.lower()
    if len(tok) < 6 or tok in NOISE_WHITELIST:
        return False
    longest_consonant_run = max(len(r) for r in re.findall(r"[^aeiouy]+", tok) or [""])
    vowel_ratio = sum(c in "aeiouy" for c in tok) / len(tok)
    return longest_consonant_run >= 5 or (len(tok) >= 8 and vowel_ratio < 0.2)


def normalise_chars(text):
    t = ftfy.fix_text(str(text))
    t = unicodedata.normalize("NFKC", t)          # fullwidth letters etc.
    t = ZW_ALWAYS_RE.sub("", t)
    t = ZWNJ_RE.sub("", t)
    t = ZWJ_RE.sub("", t)
    t = t.translate(LOOKALIKES)
    t = EMOJI_RE.sub("", t)
    t = t.replace("[.]", ".").replace("(.)", ".")
    t = re.sub(r"hxxp", "http", t, flags=re.I)
    t = REPEAT_RE.sub(r"\1\1", t)                 # freeeee -> freee
    t = re.sub(r"([^\w\s])\1{2,}", r"\1", t)      # ]]] -> ]
    t = re.sub(r"(?<=[A-Za-z])[0134578](?=[A-Za-z])",   # P1N -> PIN, k0tha -> kotha
               lambda m: {"0": "o", "1": "i", "3": "e", "4": "a",
                          "5": "s", "7": "t", "8": "b"}[m.group()], t)
    return t


def clean_message(msg):
    """One message -> (speaker or None, cleaned text, n_noise_tokens)."""
    t = normalise_chars(msg)
    t = URL_RE.sub(" URL ", t)
    speaker = None
    m = SPEAKER_RE.match(t)
    if m:
        speaker = m.group(1).upper()
        t = t[m.end():]
    n_noise = 0
    def _drop(match):
        nonlocal n_noise
        if is_noise_token(match.group(0)):
            n_noise += 1
            return " "
        return match.group(0)
    t = WORD_RE.sub(_drop, t)
    t = re.sub(r"\s+", " ", t).strip()
    return speaker, t, n_noise


def preprocess(text):
    """Raw record -> dict with the cleaned conversation and a few flags."""
    raw_msgs = [m for m in str(text).split(config.DELIMITER)]
    parsed = [clean_message(m) for m in raw_msgs]
    parsed = [p for p in parsed if p[1] or p[0]] or [(None, "", 0)]
    parts = [f"{sp}: {tx}" if sp else tx for sp, tx, _ in parsed]
    clean = f" {config.DELIMITER} ".join(parts)
    body = " ".join(tx for _, tx, _ in parsed)
    has_letters = bool(re.search(rf"[A-Za-z{BANGLA}]{{2,}}", body))
    return {
        "clean": clean,
        "n_messages": len(parsed),
        "n_noise_tokens": sum(p[2] for p in parsed),
        "noise_only": not has_letters,           # nothing meaningful left after cleaning
        "raw_len": len(str(text)),
        "had_lookalike": bool(re.search("[\u0400-\u04FF\u0370-\u03FF]", str(text))),
        "had_zero_width": bool(ZW_ALWAYS_RE.search(str(text))),
        "had_emoji": bool(EMOJI_RE.search(str(text))),
    }


def template_key(clean_text):
    """Key for grouping/deduplicating near-identical templates (not a model input)."""
    t = clean_text.lower()
    t = re.sub(r"\d+", "0", t)
    t = re.sub(r"\b[su|r]\s*:", "", t) if False else t
    t = re.sub(r"\s+", " ", t).strip()
    return t


def preprocess_frame(df):
    feats = pd.DataFrame([preprocess(t) for t in df["text"]], index=df.index)
    feats["template_key"] = feats["clean"].map(template_key)
    return pd.concat([df, feats], axis=1)


def main():
    from src.data_loader import load_sample, load_test
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 120)

    s = preprocess_frame(load_sample())
    t = preprocess_frame(load_test())

    print("sample noise-only:", int(s.noise_only.sum()), "| test noise-only:", int(t.noise_only.sum()))
    print("test rows with noise tokens removed:", f"{(t.n_noise_tokens > 0).mean():.2%}")
    print("unique template keys in test:", t.template_key.nunique())

    print("\n--- sample before/after (rows with noise or lookalikes) ---")
    for _, r in s[(s.n_noise_tokens > 0) | s.had_lookalike | s.had_zero_width].head(8).iterrows():
        print("RAW  :", r.text[:130])
        print("CLEAN:", r.clean[:130], "\n")

    print("--- test rows flagged noise-only (check these look right) ---")
    print(t.loc[t.noise_only, "text"].head(15).to_string())
    print("\n--- random test rows with noise tokens removed ---")
    for _, r in t[t.n_noise_tokens > 0].sample(8, random_state=config.SEED).iterrows():
        print("RAW  :", r.text[:130])
        print("CLEAN:", r.clean[:130], "\n")


if __name__ == "__main__":
    main()