"""Stage 1: dataset exploration.

Run from the project root:
    python -m src.text_analysis
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.data_loader import (
    ACTION_NAMES,
    LABEL_NAMES,
    load_sample,
    load_submission_template,
    load_test,
    parse_techniques,
)

BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
SUSPICIOUS_ZW_RE = re.compile(r"[\u200b\u2060\ufeff]")  # ZWNJ/ZWJ are normal in Bangla, so excluded
SPEAKER_RE = re.compile(r"^\s*[A-Za-z]{1,10}\s*:")

BANGLISH_WORDS = {
    "apnar", "apni", "ami", "tumi", "korun", "korte", "kore", "koren", "bolen", "bolun",
    "taka", "ta", "ei", "eta", "ekta", "ase", "ache", "hoyeche", "hobe", "dite", "din",
    "jan", "pathan", "pathaisi", "ferot", "bhul", "vul", "kauke", "bolben", "na", "nai",
    "sob", "shob", "jitechen", "jiteche", "income", "ghor", "bose", "dhonyobad", "meyad",
    "shortto", "projojjo", "kinle", "dhonnobad", "bhai", "apu", "lagbe", "ki", "ekhon",
    "bkash", "nagad", "rocket", "upay", "shuru", "ekhoni", "joldi", "onek",
}


def language_bucket(text):
    has_bn = bool(BANGLA_RE.search(text))
    has_lat = bool(LATIN_RE.search(text))
    if has_bn and has_lat:
        return "mixed_bn_en"
    if has_bn:
        return "bangla_script"
    if not has_lat:
        return "no_letters"
    tokens = re.findall(r"[a-z]+", text.lower())
    hits = sum(t in BANGLISH_WORDS for t in tokens)
    return "banglish" if hits >= 2 else "english"


def norm_key(text):
    """Aggressive key used only to detect duplicates (never used as model input)."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = SUSPICIOUS_ZW_RE.sub("", t)
    t = re.sub(r"\d+", "0", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def add_text_stats(df):
    df = df.copy()
    t = df["text"]
    df["n_messages"] = t.str.count(re.escape(config.DELIMITER)) + 1
    df["is_conversation"] = df["n_messages"] > 1
    df["has_speaker_marker"] = t.map(lambda x: bool(SPEAKER_RE.search(x)))
    df["char_len"] = t.str.len()
    df["language"] = t.map(language_bucket)
    df["has_cyrillic"] = t.map(lambda x: bool(CYRILLIC_RE.search(x)))
    df["has_zero_width"] = t.map(lambda x: bool(SUSPICIOUS_ZW_RE.search(x)))
    df["has_url"] = t.str.contains(r"https?://|hxxps?://|\[\.\]", regex=True)
    df["has_emoji"] = t.map(lambda x: any(ord(c) >= 0x1F300 for c in x))
    df["norm_key"] = t.map(norm_key)
    return df


def near_duplicate_pairs(texts, threshold=0.85):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)
    ii, jj = np.where(np.triu(sim) >= threshold)
    return [(int(i), int(j), float(sim[i, j])) for i, j in zip(ii, jj)]


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 110)

    sample = load_sample()
    test = load_test()
    template = load_submission_template()

    section("1. FILES, SHAPES, COLUMNS, DTYPES")
    print("sample:", sample.shape, "| test:", test.shape, "| submission template:", template.shape)
    print(sample.dtypes, "\n")
    print("missing/empty text -> sample:", (sample.text.str.strip() == "").sum(),
          "| test:", (test.text.str.strip() == "").sum())
    print("test ids unique:", test.id.is_unique,
          "| test ids == template ids (same order):", bool((test.id.values == template.id.values).all()))

    section("2. CLASS / ACTION / TECHNIQUE COUNTS (sample)")
    print(sample.label.map(LABEL_NAMES).value_counts(), "\n")
    print(sample.action.map(ACTION_NAMES).value_counts(), "\n")
    tech = sample.techniques.map(parse_techniques)
    print("technique frequency:")
    print(tech.explode().fillna("NONE").value_counts(), "\n")
    print("label x action:")
    print(pd.crosstab(sample.label, sample.action))
    print("\nlabel x number of techniques:")
    print(pd.crosstab(sample.label, tech.map(len)))

    s = add_text_stats(sample)
    t = add_text_stats(test)

    section("3. LANGUAGE MIX")
    print("sample:\n", s.language.value_counts(normalize=True).round(3), "\n")
    print("test:\n", t.language.value_counts(normalize=True).round(3))
    print("\nsample label x language:")
    print(pd.crosstab(s.label, s.language))

    section("4. SINGLE MESSAGES vs CONVERSATIONS")
    print("sample conversations:", int(s.is_conversation.sum()), "of", len(s))
    print("test conversations:", int(t.is_conversation.sum()), "of", len(t))
    print("\ntest n_messages distribution:")
    print(t.n_messages.value_counts().sort_index())
    print("\nsample label x is_conversation:")
    print(pd.crosstab(s.label, s.is_conversation))
    print("\nconversations with speaker markers -> sample:",
          int((s.is_conversation & s.has_speaker_marker).sum()),
          "| test:", int((t.is_conversation & t.has_speaker_marker).sum()))

    section("5. NOISE / OBFUSCATION INDICATORS")
    for name, d in (("sample", s), ("test", t)):
        print(f"{name}: cyrillic lookalikes={d.has_cyrillic.mean():.3%} | "
              f"zero-width chars={d.has_zero_width.mean():.3%} | "
              f"urls={d.has_url.mean():.3%} | emojis={d.has_emoji.mean():.3%}")
    print("\ntext length (chars):")
    print(pd.DataFrame({"sample": s.char_len.describe(), "test": t.char_len.describe()}).round(1))

    section("6. DUPLICATES")
    print("sample exact duplicates:", int(sample.text.duplicated().sum()))
    print("sample duplicates after normalisation:", int(s.norm_key.duplicated().sum()))
    print("test exact duplicates:", int(test.text.duplicated().sum()),
          f"({test.text.duplicated().mean():.2%})")
    print("test duplicates after normalisation:", int(t.norm_key.duplicated().sum()),
          f"({t.norm_key.duplicated().mean():.2%})")
    print("unique normalised test templates:", t.norm_key.nunique())

    in_test = set(t.norm_key)
    overlap = s.norm_key.isin(in_test).sum()
    print(f"sample rows whose normalised text also appears in test: {overlap} of {len(s)}")

    pairs = near_duplicate_pairs(sample.text.tolist(), threshold=0.85)
    print(f"\nnear-duplicate pairs inside sample (char TF-IDF cosine >= 0.85): {len(pairs)}")
    for i, j, sim in pairs[:15]:
        print(f"  {sim:.2f}  [{sample.id[i]}] {sample.text[i][:60]!r}  <->  [{sample.id[j]}] {sample.text[j][:60]!r}")

    section("7. EXAMPLES: ONE PER CLASS (shortest and longest)")
    for lab in sorted(sample.label.unique()):
        sub = sample[sample.label == lab]
        print(f"\n[{lab}] {LABEL_NAMES[lab]}")
        print("  short:", sub.loc[sub.text.str.len().idxmin(), "text"][:150])
        print("  long :", sub.loc[sub.text.str.len().idxmax(), "text"][:150])


if __name__ == "__main__":
    main()