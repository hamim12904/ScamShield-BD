"""Stage 2: normalisation, noise handling, conversation parsing, and multi-view builder.

Run from the project root:
    python -m src.preprocessing
"""
from collections import Counter
from functools import lru_cache
import math
import re
import time
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
BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

BANGLA = "\u0980-\u09FF"
BANGLA_CHAR_RE = re.compile(rf"[{BANGLA}]")
LOOKALIKE_DETECT_RE = re.compile("[\u0400-\u04FF\u0370-\u03FF]")
ZW_ALWAYS_RE = re.compile("[\u200b\u2060\ufeff\u00ad]")
# ZWNJ is noise; ZWJ is kept only inside র + ZWJ + ্
ZWNJ_RE = re.compile("\u200c")
ZWJ_RE = re.compile("\u200d(?!্)|(?<!র)\u200d")
EMOJI_UNIT_RE = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF]\ufe0f?")
EMOJI_RE = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE00-\uFE0F]")
LEADING_EMOJI_RE = re.compile(r"^(?:[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE00-\uFE0F]+\s*)+")
MOJIBAKE_OR_ENTITY_RE = re.compile(r"[ÃÂâð\ufffd]|&(?:amp|lt|gt|quot|apos|nbsp|#\d+);")

# Preserve masked 'x'/'X' sequences (e.g. 01XXXXXXXXX), collapse stretched 'x' inside words
REPEAT_RE = re.compile(r"(?![xX])([^\W\d_])\1{2,}|(?<=[A-WYZa-wyz])([xX])\2{2,}(?=[A-WYZa-wyz])")
PUNCT_REPEAT_RE = re.compile(r"([^\w\s|])\1{2,}")
WS_RE = re.compile(r"\s+")
DIGITS_RE = re.compile(r"\d+")
BANGLA_OR_DIGIT_RE = re.compile(rf"[{BANGLA}0-9]")
HAS_LETTERS_RE = re.compile(rf"[A-Za-z{BANGLA}]{{2,}}")
ANY_WORD_TOKEN_RE = re.compile(rf"[A-Za-z0-9{BANGLA}]+")

# Skeleton & digit-shape helpers (strict boundaries avoid splitting 40K, 10GB, 017XXXXXX73)
_SPACED_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z][ .\-_]){3,}[A-Za-z](?![A-Za-z0-9])")
_SPACED_SEP_RE = re.compile(r"[ .\-_]")
_NUM_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])")
_LOWER_N_TAG_RE = re.compile(r"<n(\d+)>")
_OTP_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")

HXXP_RE = re.compile(r"hxxp(s?)://", re.I)
BRACKET_DOT_RE = re.compile(r"\[+\.+\]+|\(+\.+\)+")
EMPTY_BRACKET_DOT_RE = re.compile(r"(?<=[A-Za-z0-9-])\[+\.?\]+(?=[A-Za-z0-9-])")

# Lookahead (?=[a-z]) prevents testing all 22 TLDs at sentence-ending periods
URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']+"
    r"|(?<![a-z0-9-])[a-z0-9-]+\.(?=[a-z])(?:com|net|org|bd|co|io|xyz|link|info|top|vip|me|ly|site|click|online|shop|store|live|club|app|icu)\b[^\s<>\"']*",
    re.I,
)

SPEAKER_RE = re.compile(r"^\s*[\[\(]?([A-Za-z]{1,10})[\]\)]*\s*:\s*")
SPEAKER_CANON = {
    "S": "S", "R": "R", "A": "A", "B": "B",
    "SENDER": "SENDER", "YOU": "YOU", "YO": "YOU",
    "CALLER": "CALLER", "RECEIVER": "RECEIVER", "USER": "USER", "AGENT": "AGENT",
}

WORD_RE = re.compile(r"[A-Za-z]+")
ALNUM_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}
LEET_INNER_RUN_RE = re.compile(r"(?<=[A-Za-z])[0134578]+(?=[A-Za-z])")
LEET_ANY_RE = re.compile(r"[0134578]")
NON_LEET_DIGIT_RE = re.compile(r"[269]")
UNIT_OR_ID_RE = re.compile(
    r"^(?:\d+(?:gb|mb|kb|tb|k|m|b|tk|bdt|hrs?|mins?|sec|days?|st|nd|rd|th|x)"
    r"|(?:tk|bdt|rs|usd|ec|id|txn|trx|ref|ord|inv|no)\d+)$",
    re.I,
)
LEET_EXACT_MAP = {
    "m3": "me", "fr33": "free", "0tp": "otp", "p1n": "pin", "n1d": "nid",
    "5cam": "scam", "1ink": "link", "b4nk": "bank", "c4sh": "cash",
    "c0d3": "code", "pr1z3": "prize", "cl41m": "claim", "sh4r3": "share",
}

# Known noise tokens ONLY from the problem statement and sample.csv (RULES A6 compliant)
NOISE_VOCAB = {
    "wkrjfejejrn",  # problem statement §7
    "xkcdpl",       # sample.csv id 10022
    "asdkjh",       # sample.csv id 10033
    "zzqwpo",       # sample.csv id 10038
    "mnbvcx",       # sample.csv id 10053
    "lkjsdfoiu",    # sample.csv id 10058
    "xcvbnm",       # sample.csv id 10058
    "zzqpwoeiru",   # sample.csv id 10081
    "qzxvbn",       # sample.csv id 10082
}

# Domain/merged fragments that must never be dropped as noise
_KEEP_SUBSTRINGS = (
    "bkash", "bksh", "nagad", "mcash", "rocket", "upay", "paperfly", "dhl",
    "kyc", "otp", "pin", "cvv", "nid", "phone", "month", "crypt", "track",
    "claim", "thank", "passw", "cash", "card", "fraud", "bank", "taka",
    "offer", "spring", "instant", "transfer", "salary", "commiss",
)

# Rare consonant clusters for short gibberish-only detection (allows standard endings like -nks/-rds)
_GIBBERISH_CLUSTER_RE = re.compile(r"q(?!u)|[bcdfghjklmnpqrstvwxz]{4,}|^[bcdfghjklmnpqrstvwxz]{3,}")
_STD_ENDING_RE = re.compile(r"(?:nks|rds|sts|nds|nts|ths|rts|cts|pts|fts|sks|ght)$")
_COMMON_SHORT = {
    "ok", "hi", "ki", "na", "no", "yes", "call", "pay", "pin", "otp", "din", "jan",
    "mom", "bhai", "thanks", "bkash", "regards", "hello", "sorry", "please", "urgent",
}

VIEW_CONTRACT_COLS = (
    "text_raw", "text_norm", "text_skel", "n_msgs",
    "first_msg", "last_msg", "mid_msgs",
    "first_msg_len", "mean_msg_len", "max_msg_len",
    "has_url", "n_urls",
    "digit_ratio", "upper_ratio", "script_latin_ratio", "script_bangla_ratio",
    "emoji_count", "excl_count", "qmark_count", "asks_question",
    "has_otp_like_number", "noise_ratio",
)


def _is_missing(text):
    if text is None:
        return True
    if isinstance(text, float) and math.isnan(text):
        return True
    return str(text).strip() == ""


def to_skeleton(clean: str) -> str:
    """Cleaned text -> lowercase skeleton with spaced letters re-joined and digit runs as <N#>."""
    if not clean:
        return ""
    t = _SPACED_RE.sub(lambda m: _SPACED_SEP_RE.sub("", m.group(0)), clean)
    t = t.lower()
    # Keep existing <N#> tags uppercase so to_skeleton is strictly idempotent
    t = _LOWER_N_TAG_RE.sub(r"<N\1>", t)
    return _NUM_RE.sub(lambda m: f"<N{len(m.group(0))}>", t)


def _normalise_url_obfuscation(text):
    if "hxxp" in text.lower():
        text = HXXP_RE.sub(r"http\1://", text)
    if "[" in text or "(" in text:
        text = BRACKET_DOT_RE.sub(".", text)
        text = EMPTY_BRACKET_DOT_RE.sub(".", text)
    return text


def _decode_leet_token(match):
    tok = match.group(0)
    if tok.isalpha() or tok.isdigit():
        return tok
    low = tok.lower()
    if low in LEET_EXACT_MAP:
        repl = LEET_EXACT_MAP[low]
        return repl.upper() if tok.isupper() else repl
    if "xxx" in low or UNIT_OR_ID_RE.match(tok):
        return tok
    if not LEET_ANY_RE.search(tok):
        return tok
    # Case-independent guard for TrxIDs / reference codes (e.g. 9K4L2M1X / 9k4l2m1x, A4B1C7)
    if NON_LEET_DIGIT_RE.search(tok):
        return tok
    transitions = sum(tok[i].isdigit() != tok[i - 1].isdigit() for i in range(1, len(tok)))
    if transitions >= 4:
        return tok
    n_digits = sum(c.isdigit() for c in tok)
    if tok.isupper() and n_digits >= 2 and not tok.startswith(("P1N", "0TP")):
        return tok

    def _replace_match(m):
        out = []
        s = m.group(0)
        start = m.start()
        prev_up = start > 0 and tok[start - 1].isupper()
        next_up = m.end() < len(tok) and tok[m.end()].isupper()
        all_up = tok.isupper() or (prev_up and next_up)
        for ch in s:
            dec = LEET_MAP[ch]
            out.append(dec.upper() if all_up else dec)
        return "".join(out)

    if LEET_INNER_RUN_RE.search(tok):
        return LEET_ANY_RE.sub(_replace_match, tok)
    return tok


@lru_cache(maxsize=100_000)
def _is_noise_lower(t, use_vocab=True):
    if use_vocab and t in NOISE_VOCAB:
        return True
    if set(t) == {"x"}:
        return False

    n = len(t)
    vowels_y = 0
    run = 0
    longest_run = 0
    for ch in t:
        if ch in "aeiouy":
            vowels_y += 1
            run = 0
        else:
            run += 1
            if run > longest_run:
                longest_run = run

    ratio = vowels_y / n
    is_suspect = (
        vowels_y == 0
        or (n >= 8 and (longest_run >= 7 or (longest_run >= 6 and ratio < 0.20)))
    )
    if not is_suspect:
        return False
    return not any(k in t for k in _KEEP_SUBSTRINGS)


def is_noise_token(tok, use_vocab=True):
    """Known noise strings, 0-vowel >=6 char consonant strings, or long gibberish tokens."""
    if len(tok) < 6:
        return False
    return _is_noise_lower(tok.lower(), use_vocab=use_vocab)


def _is_gibberish_token(w):
    """True only for clearly unpronounceable/gibberish Latin tokens (protects Thanks/Bkash/Regards)."""
    if len(w) < 4:
        return False
    vowels = sum(c in "aeiouy" for c in w)
    if vowels == 0:
        return True
    if any(k in w for k in _KEEP_SUBSTRINGS):
        return False
    standard_ending = bool(_STD_ENDING_RE.search(w))
    if not standard_ending and _GIBBERISH_CLUSTER_RE.search(w):
        return True
    if not standard_ending and re.search(r"[bcdfghjklmnpqrstvwxz]{3,}$", w):
        return True
    return len(w) >= 7 and not standard_ending and (vowels / len(w)) < 0.18


def _is_gibberish_only(body, n_noise_removed=0):
    """Conservative whole-record check for short random-letter noise rows without Bangla, digits, or URLs."""
    if len(body) > 90 or body.count(" ") > 5:
        return False
    if BANGLA_OR_DIGIT_RE.search(body) or (("." in body or "://" in body) and URL_RE.search(body)):
        return False
    tokens = WORD_RE.findall(body.lower())
    if not tokens or len(tokens) > 3:
        return False
    if any(w in _COMMON_SHORT for w in tokens):
        return False
    if n_noise_removed > 0 and len(tokens) <= 2 and all(_is_gibberish_token(w) for w in tokens):
        return True
    suspicious = sum(_is_gibberish_token(w) for w in tokens)
    if suspicious == len(tokens):
        return True
    return suspicious > 0 and max(len(w) for w in tokens) >= 7 and len(tokens) <= 2


def _normalise_body_core(t, strip_emojis=True):
    """Internal core normalisation returning (text_with_url_placeholders, urls_list)."""
    if not t.isascii():
        if MOJIBAKE_OR_ENTITY_RE.search(t):
            t = ftfy.fix_text(t)
        t = unicodedata.normalize("NFKC", t)
        t = ZW_ALWAYS_RE.sub("", t)
        t = ZWNJ_RE.sub("", t)
        t = ZWJ_RE.sub("", t)
        t = t.translate(LOOKALIKES)
        t = t.translate(BANGLA_DIGITS)
        if strip_emojis:
            t = EMOJI_RE.sub(" ", t)
    elif "&" in t and MOJIBAKE_OR_ENTITY_RE.search(t):
        t = ftfy.fix_text(t)

    t = _normalise_url_obfuscation(t)

    urls = []
    if "." in t or "://" in t:
        def _stash_url(match):
            urls.append(match.group(0))
            return f" __URL{len(urls) - 1}__ "
        t = URL_RE.sub(_stash_url, t)

    if len(t) >= 3:
        t = REPEAT_RE.sub(lambda m: (m.group(1) or m.group(2)) * 2, t)
        t = PUNCT_REPEAT_RE.sub(r"\1", t)

    if LEET_ANY_RE.search(t):
        t = ALNUM_TOKEN_RE.sub(_decode_leet_token, t)

    return t, urls


def normalise_chars(text, strip_emojis=True):
    """Careful text normalisation preserving masked X-runs, IDs, URLs, Bangla ZWJ, and ASCII digits."""
    if _is_missing(text):
        return ""
    t, urls = _normalise_body_core(str(text), strip_emojis=strip_emojis)
    for idx, url in enumerate(urls):
        t = t.replace(f"__URL{idx}__", url)
    return WS_RE.sub(" ", t).strip()


def _strip_leading_noise_and_speaker(text):
    """Strip leading noise tokens (and leading emojis) so turn-initial speakers are always parsed."""
    n_noise = 0
    lead_emojis = []
    t = text.lstrip()
    while True:
        em = LEADING_EMOJI_RE.match(t)
        if em:
            lead_emojis.append(em.group(0).strip())
            t = t[em.end():].lstrip()
            continue
        m = WORD_RE.match(t)
        if m and is_noise_token(m.group(0)) and not t[m.end():].lstrip().startswith(":"):
            n_noise += 1
            t = t[m.end():].lstrip()
            continue
        break

    speaker = None
    sm = SPEAKER_RE.match(t)
    if sm:
        canon = SPEAKER_CANON.get(sm.group(1).upper())
        if canon:
            speaker = canon
            t = t[sm.end():]
    return speaker, t, n_noise, lead_emojis


@lru_cache(maxsize=250_000)
def _clean_turn(msg):
    """Single-pass cached turn cleaner -> (speaker, clean_tx, preserved_tx, n_noise, n_raw_tokens)."""
    t, urls = _normalise_body_core(msg, strip_emojis=False)
    speaker, t, lead_noise, lead_emojis = _strip_leading_noise_and_speaker(t)
    n_noise = lead_noise
    n_raw_tokens = len(ANY_WORD_TOKEN_RE.findall(t)) + lead_noise

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

    if lead_emojis:
        pres_tx = WS_RE.sub(" ", " ".join(lead_emojis) + " " + t).strip()
    else:
        pres_tx = WS_RE.sub(" ", t).strip()

    if t.isascii():
        clean_tx = WS_RE.sub(" ", t).strip()
    else:
        clean_tx = WS_RE.sub(" ", EMOJI_RE.sub(" ", t)).strip()

    return speaker, clean_tx, pres_tx, n_noise, n_raw_tokens


def clean_message(msg, strip_emojis=True):
    """One message -> (speaker or None, cleaned text, n_noise_tokens)."""
    if _is_missing(msg):
        return None, "", 0
    speaker, clean_tx, pres_tx, n_noise, _ = _clean_turn(str(msg))
    return speaker, (clean_tx if strip_emojis else pres_tx), n_noise


def preprocess(text):
    """Raw record -> dict with cleaned views, skeleton, conversation turns, and structural features."""
    if _is_missing(text):
        return {
            "clean": "",
            "preserved": "",
            "skel": "",
            "first_msg": "",
            "last_msg": "",
            "n_messages": 0,
            "n_noise_tokens": 0,
            "noise_ratio": 1.0,
            "noise_only": True,
            "raw_len": 0,
            "emoji_count": 0,
            "has_url": False,
            "n_urls": 0,
            "script_bangla_ratio": 0.0,
            "had_lookalike": False,
            "had_zero_width": False,
            "had_emoji": False,
        }

    raw = str(text)
    if raw.isascii():
        emoji_count = 0
        had_emoji = False
        had_lookalike = False
        had_zero_width = False
    else:
        emoji_count = len(EMOJI_UNIT_RE.findall(raw))
        had_emoji = emoji_count > 0 or bool(EMOJI_RE.search(raw))
        had_lookalike = bool(LOOKALIKE_DETECT_RE.search(raw))
        had_zero_width = bool(ZW_ALWAYS_RE.search(raw))

    raw_msgs = raw.split(config.DELIMITER)
    parsed = [_clean_turn(m) for m in raw_msgs]

    clean_parts = []
    pres_parts = []
    body_parts = []
    n_noise_tokens = 0
    n_raw_tokens = 0

    for sp, clean_tx, pres_tx, n_noise, turn_tokens in parsed:
        n_noise_tokens += n_noise
        n_raw_tokens += turn_tokens
        if clean_tx:
            clean_parts.append(f"{sp}: {clean_tx}" if sp else clean_tx)
            body_parts.append(clean_tx)
        if had_emoji and pres_tx:
            pres_parts.append(f"{sp}: {pres_tx}" if sp else pres_tx)

    delim = f" {config.DELIMITER} "
    clean = delim.join(clean_parts)
    preserved = delim.join(pres_parts) if had_emoji else clean
    skel = to_skeleton(clean)
    body = " ".join(body_parts)
    n_messages = len(raw_msgs)
    first_msg = clean_parts[0] if clean_parts else ""
    last_msg = clean_parts[-1] if clean_parts else ""

    has_letters = bool(HAS_LETTERS_RE.search(body))
    noise_only = (not has_letters) or _is_gibberish_only(body, n_noise_tokens)
    noise_ratio = 1.0 if noise_only else min(1.0, n_noise_tokens / max(n_raw_tokens, 1))

    n_urls = len(URL_RE.findall(clean)) if ("." in clean or "://" in clean) else 0
    if body:
        n_bn = len(BANGLA_CHAR_RE.findall(body))
        script_bangla_ratio = round(n_bn / max(len(body), 1), 4)
    else:
        script_bangla_ratio = 0.0

    return {
        "clean": clean,
        "preserved": preserved,
        "skel": skel,
        "first_msg": first_msg,
        "last_msg": last_msg,
        "n_messages": n_messages,
        "n_noise_tokens": n_noise_tokens,
        "noise_ratio": float(noise_ratio),
        "noise_only": noise_only,
        "raw_len": len(raw),
        "emoji_count": int(emoji_count),
        "has_url": n_urls > 0,
        "n_urls": int(n_urls),
        "script_bangla_ratio": float(script_bangla_ratio),
        "had_lookalike": had_lookalike,
        "had_zero_width": had_zero_width,
        "had_emoji": had_emoji,
    }


def template_key(clean_text):
    """Key for grouping/deduplicating near-identical templates (not a model input)."""
    t = str(clean_text).lower()
    t = DIGITS_RE.sub("0", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def preprocess_frame(df, log_every=0):
    raw_texts = [("" if _is_missing(t) else str(t)) for t in df["text"]]
    uniq_texts = list(dict.fromkeys(raw_texts))
    cache = {}
    total = len(uniq_texts)
    for i, t in enumerate(uniq_texts, 1):
        res = preprocess(t)
        res["template_key"] = template_key(res["clean"])
        cache[t] = res
        if log_every and i % log_every == 0:
            print(f"  preprocessed {i:,}/{total:,} unique rows...", flush=True)
    feats = pd.DataFrame([cache[t] for t in raw_texts], index=df.index)
    return pd.concat([df, feats], axis=1)


def build_views(df, log_every=0):
    """ARCHITECTURE.md §4 22-column contract wrapper."""
    out = preprocess_frame(df, log_every=log_every)
    out["text_raw"] = df["text"].fillna("").astype(str)
    out["text_norm"] = out["preserved"].fillna("")
    out["text_skel"] = out["skel"].fillna("")
    out["n_msgs"] = out["n_messages"]

    msgs = out["text_norm"].str.split(r"\s*\|\|\|\s*")
    n = out["text_norm"].str.len().clip(lower=1)
    out["mid_msgs"] = msgs.map(lambda m: " ||| ".join(m[1:-1]))
    out["first_msg_len"] = out["first_msg"].str.len()
    out["mean_msg_len"] = msgs.map(lambda m: sum(map(len, m)) / len(m))
    out["max_msg_len"] = msgs.map(lambda m: max(map(len, m)))
    out["digit_ratio"] = out["text_norm"].str.count(r"\d") / n
    out["upper_ratio"] = out["text_norm"].str.count(r"[A-Z]") / n
    out["script_latin_ratio"] = out["text_norm"].str.count(r"[A-Za-z]") / n
    out["excl_count"] = out["text_raw"].str.count("!")
    out["qmark_count"] = out["text_raw"].str.count(r"\?")
    out["asks_question"] = out["qmark_count"] > 0
    out["has_otp_like_number"] = out["text_norm"].str.contains(_OTP_RE, na=False)
    return out


def audit_heuristic_noise(texts, top_k=50):
    """Return top_k tokens flagged purely by the heuristic path (excluding NOISE_VOCAB)."""
    cnt = Counter(
        w.lower()
        for x in texts
        for w in WORD_RE.findall(normalise_chars(x))
        if is_noise_token(w, use_vocab=False) and w.lower() not in NOISE_VOCAB
    )
    return cnt.most_common(top_k)


def main():
    from src.data_loader import load_sample, load_test
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 120)

    s = build_views(load_sample())
    test_df = load_test()
    t0 = time.perf_counter()
    t = build_views(test_df, log_every=100_000)
    elapsed = time.perf_counter() - t0

    print(f"500k build_views runtime: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    print("sample noise-only:", int(s.noise_only.sum()), "| test noise-only:", int(t.noise_only.sum()))
    print("test rows with noise tokens removed:", f"{(t.n_noise_tokens > 0).mean():.2%}")
    print("unique template keys in test:", t.template_key.nunique())

    print("\n--- top 50 heuristic-flagged tokens (excluding NOISE_VOCAB; hand-check precision) ---")
    print(audit_heuristic_noise(test_df["text"].head(100_000), top_k=50))

    print("\n--- sample before/after (rows with noise, lookalikes, or Bangla digits) ---")
    mask = (s.n_noise_tokens > 0) | s.had_lookalike | s.had_zero_width | s.noise_only | (s.script_bangla_ratio > 0)
    for _, r in s[mask].head(10).iterrows():
        print("RAW  :", r.text[:130])
        print("CLEAN:", r.clean[:130])
        print("SKEL :", r.skel[:130], f"| noise_ratio={r.noise_ratio:.2f} noise_only={r.noise_only}\n")


if __name__ == "__main__":
    main()
