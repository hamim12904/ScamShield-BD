"""Stage 2: normalisation, noise handling, conversation parsing.

Run from the project root:
    python -m src.preprocessing
"""
import math
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
# ZWNJ is noise; ZWJ is kept only inside র + ZWJ + ্
ZWNJ_RE = re.compile("\u200c")
ZWJ_RE = re.compile("\u200d(?!্)|(?<!র)\u200d")
EMOJI_RE = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE00-\uFE0F]")
MOJIBAKE_OR_ENTITY_RE = re.compile(r"[ÃÂâð\ufffd]|&(?:amp|lt|gt|quot|apos|nbsp|#\d+);")
# Preserve masked 'x'/'X' sequences (e.g. 01XXXXXXXXX) and digits (e.g. 5000)
REPEAT_RE = re.compile(r"(?![xX])([^\W\d_])\1{2,}")
URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']+"
    r"|\b[a-z0-9-]+\.(?:com|net|org|bd|co|io|xyz|link|info|top|vip|me|ly|site|click|online|shop|store|live|club|app|icu)\b[^\s<>\"']*",
    re.I,
)
SPEAKER_RE = re.compile(r"^\s*[\[\(]?([A-Za-z]{1,10})[\]\)]*\s*:\s*")
SPEAKER_LABELS = {"S", "R", "A", "B", "SENDER", "YOU", "CALLER", "RECEIVER", "USER", "AGENT"}
WORD_RE = re.compile(r"[A-Za-z]+")
ALNUM_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}
LEET_INNER_RE = re.compile(r"(?<=[A-Za-z])[0134578](?=[A-Za-z])")

# Known noise tokens observed in sample.csv and unlabeled test text (declare in notebook)
NOISE_VOCAB = {
    "zzqwpo", "zzqpwoeiru", "qzxvbn", "mnbvcx", "asdkjh",
    "wkrjfejejrn", "xkcdpl", "lkjsdfoiu", "xcvbnm",
}

# Rare letter clusters / keyboard-mash patterns used only for whole-message noise_only fallback
_IMPOSSIBLE_CLUSTER_RE = re.compile(r"q(?!u)|[bcdfghjklmnpqrstvwxz]{4,}|^[bcdfghjklmnpqrstvwxz]{3,}")


def _is_missing(text):
    if text is None:
        return True
    if isinstance(text, float) and math.isnan(text):
        return True
    return False


def _normalise_url_obfuscation(text):
    t = re.sub(r"hxxp(s?)://", r"http\1://", text, flags=re.I)
    t = t.replace("[.]", ".").replace("(.)", ".")
    t = re.sub(r"(?<=[A-Za-z0-9-])\[\](?=[A-Za-z0-9-])", ".", t)
    return t


def _decode_leet_token(match):
    tok = match.group(0)
    if not any(c in LEET_MAP for c in tok):
        return tok
    # Leave masked phone numbers (XXX...) and uppercase IDs like TrxID 9K4L2M1X untouched
    if "XXX" in tok.upper():
        return tok
    n_digits = sum(c.isdigit() for c in tok)
    if tok.isupper() and n_digits >= 2 and not tok.startswith(("P1N", "0TP")):
        return tok
    return LEET_INNER_RE.sub(lambda m: LEET_MAP[m.group()], tok)


def is_noise_token(tok):
    """Known noise strings, 0-vowel >=6 char consonant strings, or long low-vowel/consonant-run tokens."""
    t = tok.lower()
    if not t or set(t) == {"x"}:
        return False
    if t in NOISE_VOCAB:
        return True
    if len(t) >= 6 and not any(c in "aeiouy" for c in t):
        return True
    if len(t) < 8:
        return False
    vowel_ratio = sum(c in "aeiou" for c in t) / len(t)
    longest_run = max(len(r) for r in re.findall(r"[^aeiou]+", t) or [""])  # y counts as a consonant
    return vowel_ratio < 0.15 or longest_run >= 6


def _is_gibberish_only(body):
    """Conservative whole-record check for short random-letter noise rows without Bangla, digits, or URLs."""
    if re.search(rf"[{BANGLA}0-9]", body) or URL_RE.search(body):
        return False
    tokens = WORD_RE.findall(body.lower())
    if not tokens or len(tokens) > 3:
        return False
    if any(len(w) <= 3 for w in tokens):
        # Short real tokens like 'ok', 'hi', 'ki' shouldn't be flagged unless all tokens are suspicious
        if any(w in {"ok", "hi", "ki", "na", "no", "yes", "call", "pay", "pin", "otp", "din", "jan"} for w in tokens):
            return False
    for w in tokens:
        vowels = sum(c in "aeiouy" for c in w)
        if len(w) >= 4 and (vowels == 0 or _IMPOSSIBLE_CLUSTER_RE.search(w) or vowels / len(w) < 0.22):
            continue
        return False
    return True


def normalise_chars(text, strip_emojis=True):
    """Careful text normalisation preserving masked X-runs, IDs, URLs, and Bangla ZWJ."""
    if _is_missing(text):
        return ""
    t = str(text)
    if MOJIBAKE_OR_ENTITY_RE.search(t):
        t = ftfy.fix_text(t)
    t = unicodedata.normalize("NFKC", t)          # fullwidth letters etc.
    t = ZW_ALWAYS_RE.sub("", t)
    t = ZWNJ_RE.sub("", t)
    t = ZWJ_RE.sub("", t)
    t = t.translate(LOOKALIKES)
    if strip_emojis:
        t = EMOJI_RE.sub("", t)
    t = _normalise_url_obfuscation(t)

    # Protect URLs so repetition collapsing and leetspeak don't alter domains/paths/www.
    urls = []

    def _stash_url(match):
        urls.append(match.group(0))
        return f" __URL{len(urls) - 1}__ "

    t = URL_RE.sub(_stash_url, t)
    t = REPEAT_RE.sub(r"\1\1", t)                 # freeeee -> free, keeps 5000 and 01XXXXXXXXX
    t = re.sub(r"([^\w\s])\1{2,}", r"\1", t)      # ]]] -> ]
    t = ALNUM_TOKEN_RE.sub(_decode_leet_token, t) # P1N -> PIN, ne4r357 -> near357, keeps 9K4L2M1X
    for idx, url in enumerate(urls):
        t = t.replace(f"__URL{idx}__", url)
    return t


def _strip_leading_noise_and_speaker(text):
    """Strip leading noise tokens so turn-initial speakers after noise are still parsed."""
    n_noise = 0
    t = text.lstrip()
    while True:
        m = WORD_RE.match(t)
        if m and is_noise_token(m.group(0)) and not t[m.end():].lstrip().startswith(":"):
            n_noise += 1
            t = t[m.end():].lstrip()
            continue
        break

    speaker = None
    sm = SPEAKER_RE.match(t)
    if sm and sm.group(1).upper() in SPEAKER_LABELS:
        speaker = sm.group(1).upper()
        t = t[sm.end():]
    return speaker, t, n_noise


def clean_message(msg, strip_emojis=True):
    """One message -> (speaker or None, cleaned text, n_noise_tokens)."""
    t = normalise_chars(msg, strip_emojis=strip_emojis)

    # Protect URLs before inline noise-token removal
    urls = []

    def _stash_url(match):
        urls.append(match.group(0))
        return f"__URL{len(urls) - 1}__"

    t = URL_RE.sub(_stash_url, t)
    speaker, t, lead_noise = _strip_leading_noise_and_speaker(t)
    n_noise = lead_noise

    def _drop(match):
        nonlocal n_noise
        tok = match.group(0)
        if tok.startswith("__URL"):
            return tok
        if is_noise_token(tok):
            n_noise += 1
            return " "
        return tok

    t = WORD_RE.sub(_drop, t)
    for idx, url in enumerate(urls):
        t = t.replace(f"__URL{idx}__", url)
    t = re.sub(r"\s+", " ", t).strip()
    return speaker, t, n_noise


def preprocess(text):
    """Raw record -> dict with cleaned conversation, emoji-preserved view, and quality flags."""
    if _is_missing(text):
        return {
            "clean": "",
            "preserved": "",
            "n_messages": 0,
            "n_noise_tokens": 0,
            "noise_only": True,
            "raw_len": 0,
            "had_lookalike": False,
            "had_zero_width": False,
            "had_emoji": False,
        }

    raw = str(text)
    had_emoji = bool(EMOJI_RE.search(raw))
    raw_msgs = raw.split(config.DELIMITER)
    parsed = [clean_message(m, strip_emojis=True) for m in raw_msgs]
    parsed_nonempty = [p for p in parsed if p[1]]
    if not parsed_nonempty:
        parsed_nonempty = [(None, "", sum(p[2] for p in parsed))]

    parts = [f"{sp}: {tx}" if sp else tx for sp, tx, _ in parsed_nonempty]
    clean = f" {config.DELIMITER} ".join(parts)

    if had_emoji:
        pres_parsed = [clean_message(m, strip_emojis=False) for m in raw_msgs]
        pres_nonempty = [p for p in pres_parsed if p[1]] or [(None, "", 0)]
        preserved = f" {config.DELIMITER} ".join(
            f"{sp}: {tx}" if sp else tx for sp, tx, _ in pres_nonempty
        )
    else:
        preserved = clean

    body = " ".join(tx for _, tx, _ in parsed_nonempty)
    has_letters = bool(re.search(rf"[A-Za-z{BANGLA}]{{2,}}", body))
    noise_only = (not has_letters) or _is_gibberish_only(body)
    return {
        "clean": clean,
        "preserved": preserved,
        "n_messages": len(parsed_nonempty),
        "n_noise_tokens": sum(p[2] for p in parsed),
        "noise_only": noise_only,                # nothing meaningful left after cleaning
        "raw_len": len(raw),
        "had_lookalike": bool(re.search("[\u0400-\u04FF\u0370-\u03FF]", raw)),
        "had_zero_width": bool(ZW_ALWAYS_RE.search(raw)),
        "had_emoji": had_emoji,
    }


def template_key(clean_text):
    """Key for grouping/deduplicating near-identical templates (not a model input)."""
    t = str(clean_text).lower()
    t = re.sub(r"\d+", "0", t)
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

    print("\n--- sample before/after (rows with noise, lookalikes, or URLs) ---")
    mask = (s.n_noise_tokens > 0) | s.had_lookalike | s.had_zero_width | s.noise_only
    for _, r in s[mask].head(10).iterrows():
        print("RAW  :", r.text[:130])
        print("CLEAN:", r.clean[:130], f"| noise_only={r.noise_only}\n")

    print("--- test rows flagged noise-only (check these look right) ---")
    print(t.loc[t.noise_only, "text"].head(20).to_string())
    print("\n--- random test rows with noise tokens removed ---")
    for _, r in t[t.n_noise_tokens > 0].sample(8, random_state=config.SEED).iterrows():
        print("RAW  :", r.text[:130])
        print("CLEAN:", r.clean[:130], "\n")


if __name__ == "__main__":
    main()