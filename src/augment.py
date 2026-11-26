"""Programmatic label-preserving corruption (Set A for training, Set B held out for evaluation).

Contract (ARCHITECTURE.md §4):
    corrupt(text, rng, level="medium", ops="A") -> str
    expand(df, k=3, seed=config.SEED, level="medium", ops="A") -> DataFrame
"""
import random

import pandas as pd

from src import config

_LEVEL_STRENGTH = {
    "light": 0.04,
    "medium": 0.07,
    "heavy": 0.12,
}

# Set A (training): leet + Cyrillic subset + standard typos/spacing/ZWSP/emojis
_SET_A_LEET = {"o": "0", "i": "1", "e": "3", "a": "4", "s": "5", "t": "7", "O": "0", "I": "1", "E": "3", "A": "4", "S": "5", "T": "7"}
_SET_A_HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "i": "і"}
_SET_A_EMOJIS = ["😊", "🎁", "🔥", "💰", "✅", "📱", "🙏"]
_ASCII_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Set B (evaluation only): held-out Greek/fullwidth/extended homoglyphs, transliterations,
# word-boundary removal, ZWNJ/soft-hyphen, and held-out synthetic noise strings
_SET_B_HOMOGLYPHS = {
    "A": "Α", "B": "Β", "E": "Ε", "Z": "Ζ", "H": "Η", "I": "Ι", "K": "Κ",
    "M": "Μ", "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ", "X": "Χ",
    "o": "ο", "v": "ν", "p": "ρ", "i": "ι", "j": "ј", "s": "ѕ", "h": "һ",
    "a": "\uff41", "e": "\uff45", "t": "\uff54", "n": "\uff4e",  # fullwidth (NFKC stress)
}
_SET_B_RESPELLINGS = {
    "bhul": "vul", "vul": "bhul", "taka": "tk", "korte": "krte",
    "shob": "sob", "sob": "shob", "bolen": "bln", "ekhoni": "ekhn",
    "please": "pls", "message": "msg", "number": "nmbr", "account": "accnt",
    "verify": "vrify", "delivery": "dlvry", "free": "fre",
}
_SET_B_NOISE = ("qvrtxmnpk", "zplmnbvk", "krtwqzxp", "jhvfgbcn", "plmkqzxw", "vbnmkjhgf")
_SET_B_ZW = ("\u200c", "\u00ad", "\u2060")


def _corrupt_turn_a(text, rng, strength):
    out = []
    for ch in str(text):
        r = rng.random()
        if r < strength * 0.22:                                   # deletion
            continue
        if r < strength * 0.42:                                   # repetition
            out.append(ch * rng.randint(2, 3))
            continue
        if r < strength * 0.58 and ch in _SET_A_LEET:             # leet (e.g. Ver1fy, 0TP)
            out.append(_SET_A_LEET[ch])
            continue
        if r < strength * 0.72 and ch.isalpha():                  # random char substitution
            rep = rng.choice(_ASCII_LETTERS)
            out.append(rep.upper() if ch.isupper() else rep)
            continue
        if r < strength * 0.84 and ch in _SET_A_HOMOGLYPHS:       # Set A Cyrillic lookalike
            out.append(_SET_A_HOMOGLYPHS[ch])
            continue
        if r < strength * 0.92:                                   # ZWSP insertion
            out.append(ch + "\u200b")
            continue
        if r < strength:                                          # random case
            out.append(ch.swapcase())
            continue
        out.append(ch)

    t = "".join(out)
    if rng.random() < 0.30:
        t = t.replace(" ", "", rng.randint(1, 2))
    if rng.random() < 0.25:
        t = f"{t} {rng.choice(_SET_A_EMOJIS)}"
    if rng.random() < 0.15:
        t = f"{t}{'!' * rng.randint(2, 3)}"
    return t


def _corrupt_turn_b(text, rng, strength):
    # Transliteration-style respellings on whole words first
    words = str(text).split(" ")
    resp = []
    for w in words:
        lw = w.lower()
        if lw in _SET_B_RESPELLINGS and rng.random() < min(0.45, strength * 4):
            rep = _SET_B_RESPELLINGS[lw]
            resp.append(rep.upper() if w.isupper() else rep)
        else:
            resp.append(w)
    t = " ".join(resp)

    # Character-level held-out Greek/fullwidth homoglyphs, repetitions, ZWNJ/soft-hyphen, deletions
    out = []
    for ch in t:
        r = rng.random()
        if r < strength * 0.22:                                   # deletion
            continue
        if r < strength * 0.45:                                   # repetition
            out.append(ch * rng.randint(2, 3))
            continue
        if r < strength * 0.75 and ch in _SET_B_HOMOGLYPHS:       # Set B held-out homoglyph
            out.append(_SET_B_HOMOGLYPHS[ch])
            continue
        if r < strength * 0.88:                                   # ZWNJ / soft-hyphen / WJ
            out.append(ch + rng.choice(_SET_B_ZW))
            continue
        if r < strength:
            out.append(ch.swapcase())
            continue
        out.append(ch)
    t = "".join(out)

    # Word-boundary removal (merge adjacent words) + occasional spaced-out token
    if rng.random() < 0.40:
        t = t.replace(" ", "", rng.randint(1, 3))
    if rng.random() < 0.25:
        noise = rng.choice(_SET_B_NOISE)
        parts = t.split(" ")
        parts.insert(rng.randint(0, len(parts)), noise)
        t = " ".join(parts)
    if rng.random() < 0.20:
        parts = t.split(" ")
        long_idxs = [i for i, w in enumerate(parts) if len(w) >= 5 and w.isalpha()]
        if long_idxs:
            idx = rng.choice(long_idxs)
            sep = rng.choice([" ", "."])
            parts[idx] = sep.join(parts[idx])
            t = " ".join(parts)
    return t


def corrupt(text, rng, level="medium", ops="A"):
    """Label-preserving corruption keeping `|||` intact.

    ops='A': training operators (leet, random char sub, ZWSP, emojis, spacing).
    ops='B': held-out evaluation operators (Greek/fullwidth homoglyphs, respellings,
             word-boundary removal, ZWNJ/soft-hyphen, held-out noise strings).
    """
    strength = _LEVEL_STRENGTH.get(level, _LEVEL_STRENGTH["medium"])
    fn = _corrupt_turn_a if str(ops).upper() == "A" else _corrupt_turn_b
    turns = str(text).split(config.DELIMITER)
    corrupted = [fn(turn.strip(), rng, strength) for turn in turns]
    return f" {config.DELIMITER} ".join(corrupted)


def expand(df, k=3, seed=config.SEED, level="medium", ops="A", weight=0.75):
    """Create `k` corrupted training variants per row, inheriting `group_id`."""
    rng = random.Random(seed)
    rows = []
    has_group_id = "group_id" in df.columns
    has_group = "group" in df.columns

    for _, row in df.iterrows():
        gid = row["group_id"] if has_group_id else (row["group"] if has_group else row["id"])
        for _ in range(k):
            item = row.to_dict()
            item["text"] = corrupt(row["text"], rng, level=level, ops=ops)
            item["group_id"] = gid
            if has_group:
                item["group"] = gid
            item["source"] = "aug"
            item["weight"] = float(weight)
            rows.append(item)

    return pd.DataFrame(rows)
