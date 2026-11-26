"""LLM-based zero-shot labeling and dataset expansion for ScamShield BD.

Owned by Teammate B (Step 3).

Provider-agnostic labeling pipeline for test templates with caching, rate-limiting,
calibration gating, and consistency filtering.

Public API
----------
select_templates(test_df, n=3000, seed=config.SEED) -> pd.DataFrame
build_prompt(text: str) -> dict
label_rows(texts, cache_path=None, max_cost=5.0, client_fn=None) -> pd.DataFrame
calibrate_prompt_on_real(sample_df=None, client_fn=None) -> dict
filter_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

from src import config
from src.data_loader import load_sample, load_test, techniques_to_multihot
from src.evaluation.metrics import macro_f1_multiclass, macro_f1_techniques
from src.preprocessing import preprocess, template_key
from src.text_analysis import language_bucket

# Load credentials from .env safely without printing
load_dotenv(dotenv_path=config.ROOT / ".env")

log = logging.getLogger(__name__)

CACHE_FILE_DEFAULT = config.ROOT / "data" / "extra" / "llm_labels_v1.jsonl"

# Technique definitions from competition specifications
TECHNIQUE_DEFINITIONS = {
    "T1": "Phishing link, malicious URL, or fake portal redirecting users",
    "T2": "Credential harvesting (asking for PIN, OTP, password, card details, NID)",
    "T3": "False urgency, threat of account suspension, or immediate deadline pressure",
    "T4": "Unearned reward, prize, lottery, or false cashback lure",
    "T5": "Brand / authority impersonation (bKash, Nagad, courier, government, bank)",
    "T6": "Job lure, fake task earnings, money doubling, or bogus investment scheme",
}

# Pricing per million tokens (conservative estimate: ~$0.15 input, $0.60 output)
INPUT_TOKEN_COST_PER_M = 0.15
OUTPUT_TOKEN_COST_PER_M = 0.60


def _get_api_credentials() -> tuple[str, str, str]:
    """Retrieve LLM provider, model, and API key from environment without logging secrets."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    key = os.getenv("LLM_API_KEY", "")
    return provider, model, key


def _compute_text_hash(text: str) -> str:
    """Return deterministic SHA-256 hash of text."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


# ── 1. Template Selection ─────────────────────────────────────────────────────

def select_templates(
    test_df: pd.DataFrame,
    n: int = 3000,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Select n representative test templates stratified by language, turns, and noise ratio.

    Steps:
    1. Preprocess each record, dropping noise-only rows.
    2. Deduplicate by template_key.
    3. Stratified sample across language bucket, message count, and noise ratio.
    """
    records = []
    for _, row in test_df.iterrows():
        t = str(row.get("text", ""))
        p = preprocess(t)
        if p["noise_only"]:
            continue

        raw_text = t
        norm_text = p["preserved"]
        t_key = p.get("template_key") or template_key(p["clean"])

        lang = language_bucket(norm_text)
        n_msg = p["n_messages"]
        if n_msg <= 1:
            msg_bucket = "1"
        elif n_msg <= 4:
            msg_bucket = "2-4"
        else:
            msg_bucket = "5-12"

        nr = p["noise_ratio"]
        if nr <= 0.05:
            nr_bucket = "low"
        elif nr <= 0.20:
            nr_bucket = "mid"
        else:
            nr_bucket = "high"

        stratum = f"{lang}_{msg_bucket}_{nr_bucket}"

        records.append({
            "id": row.get("id"),
            "text": raw_text,
            "norm": norm_text,
            "template_key": t_key,
            "stratum": stratum,
        })

    if not records:
        return pd.DataFrame(columns=["id", "text", "template_key", "stratum"])

    df_prep = pd.DataFrame(records)
    # Deduplicate by template_key, keeping the first occurrence
    df_dedup = df_prep.drop_duplicates(subset=["template_key"]).reset_index(drop=True)

    if len(df_dedup) <= n:
        return df_dedup

    # Deterministic stratified sampling
    rng = np.random.RandomState(seed)
    sampled_indices = []
    strata_groups = df_dedup.groupby("stratum")
    total_len = len(df_dedup)

    for _, group in strata_groups:
        prop = len(group) / total_len
        group_n = max(1, int(round(prop * n)))
        chosen = rng.choice(group.index.values, size=min(len(group), group_n), replace=False)
        sampled_indices.extend(chosen)

    # Adjust sample size if rounding produced slight surplus or deficit
    if len(sampled_indices) > n:
        sampled_indices = rng.choice(sampled_indices, size=n, replace=False).tolist()
    elif len(sampled_indices) < n:
        remaining = list(set(df_dedup.index.values) - set(sampled_indices))
        extra_n = min(len(remaining), n - len(sampled_indices))
        if extra_n > 0:
            sampled_indices.extend(rng.choice(remaining, size=extra_n, replace=False).tolist())

    return df_dedup.loc[sampled_indices].sort_values("id").reset_index(drop=True)


# ── 2. Prompt Builder ─────────────────────────────────────────────────────────

def build_prompt(text: str) -> dict:
    """Build system and user prompts with exact intent definitions and strict JSON schema."""
    system_prompt = (
        "You are an expert fraud and security analyst classifying SMS and chat messages in Bangladesh.\n"
        "Messages may be in English, Bangla script, Banglish (Bangla in Latin letters), or mixed.\n"
        "Conversations of multiple messages are separated by '|||'.\n\n"
        "### CLASSIFICATION LABELS (Decision Priority: 1 -> 2 -> 4 -> 3 -> 5 -> 0)\n"
        "Stop at the first matching label:\n"
        "1. MFS/OTP scam: Asks for PIN, OTP, password, NID, or card details; claims a mistaken transfer "
        "and demands money back; impersonates bKash/Nagad/Rocket/bank support to gain access.\n"
        "2. Prize/lottery scam: Unearned prize, lottery, lottery draw, gift, or cashback requiring payment, "
        "clicking a link, or submitting details to claim.\n"
        "4. Delivery phishing: Unexpected parcel, courier (e.g. Steadfast, Pathao, RedX, Sundarban), "
        "customs fee, address confirmation, or tracking link phishing.\n"
        "3. Fake job/investment: Unrealistic income, work-from-home task earnings, upfront recruitment fee, "
        "money doubling, or cryptocurrency investment.\n"
        "5. Promotional spam: Marketing, discounts, or app promotions that are unsolicited/spammy but NOT fraudulent.\n"
        "0. Legitimate: Transaction confirmations, real service updates, personal chats, or official notifications.\n\n"
        "### HARD-LEGIT RULES (Must classify as Legit label 0, or Promo label 5):\n"
        "- A genuine OTP delivery that states 'Do not share your OTP/PIN with anyone' is Legit (label 0, action 0, techniques []).\n"
        "- A real delivery/courier status update (e.g. 'Your order is out for delivery') without asking for fees/links is Legit (label 0, action 0, techniques []).\n"
        "- A real bill payment reminder or account balance SMS is Legit (label 0, action 0, techniques []).\n"
        "- A friend or contact warning about an active scam is Legit (label 0, action 0, techniques []).\n"
        "- A genuine discount, promotional coupon, or telco bundle offer is Promotional spam (label 5, action 0, techniques []).\n\n"
        "### TECHNIQUE CODES (Select all that apply for scam labels 1-4; empty list [] for labels 0 and 5):\n"
        "- T1: Phishing link, malicious URL, or fake portal\n"
        "- T2: Credential harvesting (PIN/OTP/password/card/NID)\n"
        "- T3: False urgency, threats, account blocked/suspended immediately\n"
        "- T4: Unearned prize, gift, reward, or lottery lure\n"
        "- T5: Brand / authority impersonation (bKash, Nagad, bank, courier)\n"
        "- T6: Task / job earnings or investment money doubling\n\n"
        "### ACTION CODES (0 to 5):\n"
        "0: No action (Required for label 0 and label 5)\n"
        "1: Share credentials\n"
        "2: Pay / transfer money\n"
        "3: Click link\n"
        "4: Install / join / task\n"
        "5: Contact sender\n\n"
        "### OUTPUT FORMAT:\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "label": <int 0-5>,\n'
        '  "techniques": [<list of strings from "T1".."T6", or empty []>],\n'
        '  "action": <int 0-5>,\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "language": <string "english"|"banglish"|"bangla_script"|"mixed_bn_en">\n'
        "}"
    )
    user_prompt = f"Analyze and classify this single message record:\n\n{text}"
    return {
        "system": system_prompt,
        "user": user_prompt,
    }


# ── 3. Labeling Engine with Caching and Retry ─────────────────────────────────

def _execute_api_call(
    prompt: dict,
    provider: str,
    model: str,
    api_key: str,
    timeout: float = 30.0,
) -> dict:
    """Execute raw HTTP call to the configured LLM provider."""
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is not set in environment or .env. "
            "Callers should use the frozen cache file data/extra/llm_labels_v1.jsonl "
            "or set LLM_API_KEY before executing live labeling."
        )

    headers = {"Content-Type": "application/json"}

    if provider in ("openai", "openrouter", "generic"):
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
        }
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {
                "content": content,
                "prompt_tokens": usage.get("prompt_tokens", 800),
                "completion_tokens": usage.get("completion_tokens", 60),
            }

    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 512,
            "system": prompt["system"],
            "messages": [{"role": "user", "content": prompt["user"]}],
        }
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["content"][0]["text"]
            usage = data.get("usage", {})
            return {
                "content": content,
                "prompt_tokens": usage.get("input_tokens", 800),
                "completion_tokens": usage.get("output_tokens", 60),
            }

    elif provider in ("gemini", "google"):
        base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        url = f"{base_url}/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": f"{prompt['system']}\n\n{prompt['user']}"}]}],
            "generationConfig": {
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        }
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            return {
                "content": content,
                "prompt_tokens": usage.get("promptTokenCount", 800),
                "completion_tokens": usage.get("candidatesTokenCount", 60),
            }

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def label_rows(
    texts: Sequence[str],
    cache_path: Optional[Path] = None,
    max_cost: float = 5.0,
    client_fn: Optional[Callable[[dict], dict]] = None,
    delay: float = 0.05,
) -> pd.DataFrame:
    """Label text records with disk caching, resume capability, rate limiting, and cost cap."""
    cache_path = Path(cache_path or CACHE_FILE_DEFAULT)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load existing cache
    cached_records: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if "key" in obj and "response" in obj:
                            cached_records[obj["key"]] = obj["response"]
                    except json.JSONDecodeError:
                        continue

    provider, model, api_key = _get_api_credentials()
    accumulated_cost = 0.0
    results: list[dict] = []

    for idx, text in enumerate(texts):
        key = _compute_text_hash(text)

        # Check cache
        if key in cached_records:
            resp_dict = cached_records[key]
            results.append({
                "key": key,
                "text": text,
                **resp_dict,
            })
            continue

        # Check cost threshold
        if accumulated_cost >= max_cost:
            log.warning("Cost cap of $%.2f reached. Halting live labeling at %d texts.", max_cost, idx)
            break

        prompt = build_prompt(text)

        # Execute API call with retries
        success = False
        parsed_resp = None
        for attempt in range(3):
            try:
                if client_fn is not None:
                    raw_out = client_fn(prompt)
                else:
                    raw_out = _execute_api_call(prompt, provider, model, api_key)

                content = raw_out["content"] if isinstance(raw_out, dict) else str(raw_out)
                p_tokens = raw_out.get("prompt_tokens", 800) if isinstance(raw_out, dict) else 800
                c_tokens = raw_out.get("completion_tokens", 60) if isinstance(raw_out, dict) else 60

                call_cost = (p_tokens / 1_000_000 * INPUT_TOKEN_COST_PER_M) + (
                    c_tokens / 1_000_000 * OUTPUT_TOKEN_COST_PER_M
                )
                accumulated_cost += call_cost

                parsed_resp = json.loads(content)
                success = True
                break
            except Exception as e:
                log.warning("API attempt %d failed for key %s: %s", attempt + 1, key[:8], e)
                time.sleep(2.0 ** attempt)

        if success and parsed_resp is not None:
            # Append to cache
            cache_entry = {
                "key": key,
                "text": text,
                "response": parsed_resp,
                "model": model,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")

            cached_records[key] = parsed_resp
            results.append({
                "key": key,
                "text": text,
                **parsed_resp,
            })
            time.sleep(delay)
        else:
            log.error("Failed to obtain response for text key %s", key)

    return pd.DataFrame(results)


# ── 4. Calibration Gate on Real Sample ────────────────────────────────────────

def calibrate_prompt_on_real(
    sample_df: Optional[pd.DataFrame] = None,
    client_fn: Optional[Callable[[dict], dict]] = None,
    min_macro_f1: float = 0.80,
    cache_path: Optional[Path] = None,
) -> dict:
    """Label the 86 real rows of data/sample.csv and verify the prompt satisfies the quality gate.

    GATE: label Macro-F1 must be >= 0.80.
    """
    if sample_df is None:
        sample_df = load_sample()

    texts = sample_df["text"].tolist()
    y_true_a = sample_df["label"].to_numpy(dtype=int)
    Y_true_b = techniques_to_multihot(sample_df["techniques"])
    y_true_c = sample_df["action"].to_numpy(dtype=int)

    log.info("Calibrating prompt on %d labeled rows (zero-shot, text only)...", len(texts))
    df_pred = label_rows(texts, cache_path=cache_path, client_fn=client_fn)

    if len(df_pred) < len(texts):
        raise RuntimeError(
            f"Could only label {len(df_pred)} of {len(texts)} sample rows. "
            "Ensure API key is funded and valid."
        )

    y_pred_a = df_pred["label"].fillna(0).astype(int).to_numpy()
    tech_series = df_pred["techniques"].apply(
        lambda ts: "|".join(ts) if isinstance(ts, list) and ts else "NONE"
    )
    Y_pred_b = techniques_to_multihot(tech_series)
    y_pred_c = df_pred["action"].fillna(0).astype(int).to_numpy()

    f_a = macro_f1_multiclass(y_true_a, y_pred_a)
    f_b = macro_f1_techniques(Y_true_b, Y_pred_b)
    f_c = macro_f1_multiclass(y_true_c, y_pred_c)
    cm = confusion_matrix(y_true_a, y_pred_a, labels=list(range(config.NUM_CLASSES)))

    log.info("Calibration Results: Label F1=%.4f, Techniques F1=%.4f, Action F1=%.4f", f_a, f_b, f_c)
    log.info("Label Confusion Matrix:\n%s", cm)

    if f_a < min_macro_f1:
        # Identify worst confusions
        confusions = []
        for i in range(config.NUM_CLASSES):
            for j in range(config.NUM_CLASSES):
                if i != j and cm[i, j] > 0:
                    confusions.append((i, j, cm[i, j]))
        confusions.sort(key=lambda x: x[2], reverse=True)
        conf_str = ", ".join(f"True {c[0]} -> Pred {c[1]} (n={c[2]})" for c in confusions[:5])

        raise RuntimeError(
            f"CALIBRATION GATE FAILED: Label Macro-F1 {f_a:.4f} < {min_macro_f1:.2f}. "
            f"Worst confusions: {conf_str}. Prompt must be revised before labeling test templates!"
        )

    log.info("CALIBRATION GATE PASSED: Label Macro-F1 %.4f >= %.2f", f_a, min_macro_f1)
    return {
        "F_A": f_a,
        "F_B": f_b,
        "F_C": f_c,
        "confusion_matrix": cm,
    }


# ── 5. Consistency Filtering ──────────────────────────────────────────────────

def filter_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter LLM-generated records for schema validity, confidence, and rule consistency.

    Counts and reports reasons for dropped records:
    - invalid_schema: missing required fields or non-integer labels
    - low_confidence: confidence < 0.70
    - inconsistent_rules: violates cross-tab consistency (0/5 require NONE+0; 1-4 require >=1 tech + action!=0)
    - duplicate: duplicate text or hash key
    """
    reasons = {
        "invalid_schema": 0,
        "low_confidence": 0,
        "inconsistent_rules": 0,
        "duplicate": 0,
    }
    if df.empty:
        return df, reasons

    valid_indices = []
    seen_keys = set()

    for idx, row in df.iterrows():
        key = row.get("key") or _compute_text_hash(row.get("text", ""))
        if key in seen_keys:
            reasons["duplicate"] += 1
            continue

        label = row.get("label")
        action = row.get("action")
        techniques = row.get("techniques")
        conf = row.get("confidence")

        # 1. Schema check
        if (
            label is None
            or action is None
            or conf is None
            or pd.isna(label)
            or pd.isna(action)
            or pd.isna(conf)
            or not isinstance(techniques, list)
        ):
            reasons["invalid_schema"] += 1
            continue

        try:
            label_int = int(label)
            action_int = int(action)
            if float(label) != label_int or float(action) != action_int:
                reasons["invalid_schema"] += 1
                continue
            if not (0 <= label_int <= 5 and 0 <= action_int <= 5):
                reasons["invalid_schema"] += 1
                continue
        except (ValueError, TypeError):
            reasons["invalid_schema"] += 1
            continue

        # 2. Confidence check
        try:
            conf_val = float(conf)
            if conf_val < 0.70:
                reasons["low_confidence"] += 1
                continue
        except (ValueError, TypeError):
            reasons["invalid_schema"] += 1
            continue

        # 3. Consistency rules
        # Labels 0 and 5 must have no techniques and action == 0
        if label_int in (0, 5):
            if len(techniques) > 0 or action_int != 0:
                reasons["inconsistent_rules"] += 1
                continue
        # Labels 1 to 4 must have at least one technique and action != 0
        elif label_int in (1, 2, 3, 4):
            if len(techniques) == 0 or action_int == 0:
                reasons["inconsistent_rules"] += 1
                continue

        seen_keys.add(key)
        valid_indices.append(idx)

    filtered_df = df.loc[valid_indices].reset_index(drop=True)
    return filtered_df, reasons


# ── 6. CLI Execution ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-based test template labeling.")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration gate on real sample")
    parser.add_argument("--label", action="store_true", help="Label selected test templates")
    parser.add_argument("--n", type=int, default=3000, help="Number of test templates to sample (default: 3000)")
    parser.add_argument("--max-cost", type=float, default=5.0, help="Maximum allowed API cost in USD (default: 5.0)")
    parser.add_argument("--cache-path", type=str, default=str(CACHE_FILE_DEFAULT), help="Path to jsonl cache file")
    args = parser.parse_args()

    if args.calibrate:
        res = calibrate_prompt_on_real()
        print(f"Calibration successful: Label F1 = {res['F_A']:.4f}")
    elif args.label:
        test_df = load_test()
        print(f"Loaded {len(test_df)} test rows. Selecting up to {args.n} templates...")
        templates = select_templates(test_df, n=args.n)
        print(f"Selected {len(templates)} diverse templates.")
        labeled_df = label_rows(templates["text"].tolist(), cache_path=Path(args.cache_path), max_cost=args.max_cost)
        filtered_df, reasons = filter_labels(labeled_df)
        print(f"Labeling complete. Raw: {len(labeled_df)}, Filtered: {len(filtered_df)}")
        print("Dropped reasons:", reasons)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
