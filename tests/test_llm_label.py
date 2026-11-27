"""Tests for src/llm_label.py using mocked clients (no network)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src import llm_label
from src.llm_label import (
    TECHNIQUE_DEFINITIONS,
    _gate_passed,
    _parse_response,
    _RateLimited,
    _redact,
    build_prompt,
    calibrate_prompt_on_real,
    filter_labels,
    label_rows,
    prompt_version,
    select_templates,
)

GOOD = {"label": 1, "techniques": ["T2", "T5"], "action": 1, "confidence": 0.95, "language": "banglish"}


def _ok_client(calls=None, payload=None):
    def client(prompt):
        if calls is not None:
            calls.append(prompt["user"])
        return {"content": json.dumps(payload or GOOD), "prompt_tokens": 100, "completion_tokens": 20}
    return client


# ── prompt ────────────────────────────────────────────────────────────────────

def test_build_prompt():
    prompt = build_prompt("Apnar bKash account suspend kora hoyeche. Call 01700000000")
    assert "system" in prompt and "user" in prompt
    assert "1 -> 2 -> 4 -> 3 -> 5 -> 0" in prompt["system"]
    assert "genuine otp" in prompt["system"].lower()
    assert "T1" in prompt["system"] and "T6" in prompt["system"]
    assert "bKash" in prompt["user"]


def test_technique_definitions_match_official_problem_statement():
    # Official NeuraHack statement section 2.2
    assert "impersonation" in TECHNIQUE_DEFINITIONS["T1"].lower()
    assert "credential" in TECHNIQUE_DEFINITIONS["T2"].lower()
    assert "advance-fee" in TECHNIQUE_DEFINITIONS["T3"].lower()
    assert "prize" in TECHNIQUE_DEFINITIONS["T4"].lower()
    assert "urgency" in TECHNIQUE_DEFINITIONS["T5"].lower()
    assert "task-based" in TECHNIQUE_DEFINITIONS["T6"].lower()
    system = build_prompt("x")["system"]
    for code, definition in TECHNIQUE_DEFINITIONS.items():
        assert f"{code}: {definition}" in system


def test_prompt_has_no_sample_rows():
    sample = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "sample.csv")
    system = build_prompt("x")["system"]
    for text in sample["text"]:
        assert text not in system


# ── template selection ────────────────────────────────────────────────────────

def _test_df():
    return pd.DataFrame({
        "id": range(1, 11),
        "text": [
            "Your order is out for delivery",
            "Win $500 lottery right now",
            "Win $500 lottery right now",  # duplicate template
            "Apnar Nagad account verify korun",
            "Fake job offer part time $50 per day",
            "OTP is 123456. Do not share.",
            "Special 50% discount on recharge",
            "Urgent: your account will be suspended",
            "Short msg",
            "Another short msg",
        ],
    })


def test_select_templates_deterministic():
    test_df = _test_df()
    s1 = select_templates(test_df, n=4, seed=42)
    s2 = select_templates(test_df, n=4, seed=42)
    pd.testing.assert_frame_equal(s1, s2)
    assert len(s1) <= 4
    assert s1["template_key"].nunique() == len(s1)


def test_select_templates_pool_is_deterministic_and_dedups():
    test_df = _test_df()
    a = select_templates(test_df, n=50, seed=7, pool=6)
    b = select_templates(test_df, n=50, seed=7, pool=6)
    pd.testing.assert_frame_equal(a, b)
    assert a["template_key"].is_unique
    assert len(a) <= 6


# ── parsing ───────────────────────────────────────────────────────────────────

def test_parse_response_normalises_and_validates():
    out = _parse_response('```json\n{"label": 3, "techniques": ["t6", "T1"], "action": 4, "confidence": 0.8}\n```')
    assert out["label"] == 3 and out["techniques"] == ["T1", "T6"] and out["action"] == 4
    assert _parse_response('{"label":0,"techniques":"NONE","action":0,"confidence":0.9}')["techniques"] == []
    for bad in (
        "not json",
        '{"label": 9, "techniques": [], "action": 0, "confidence": 0.9}',
        '{"label": 1, "techniques": ["T9"], "action": 1, "confidence": 0.9}',
        '{"label": 1, "techniques": ["T1"], "action": 1, "confidence": 1.7}',
        '{"label": 1.5, "techniques": ["T1"], "action": 1, "confidence": 0.9}',
        '{"label": 1, "techniques": ["T1"], "confidence": 0.9}',
    ):
        with pytest.raises(ValueError):
            _parse_response(bad)


# ── labeling engine ───────────────────────────────────────────────────────────

def test_label_rows_caching_and_resume(tmp_path: Path):
    cache_file = tmp_path / "llm_cache.jsonl"
    calls: list = []
    texts = ["Message one", "Message two"]

    df1 = label_rows(texts, cache_path=cache_file, client_fn=_ok_client(calls))
    assert len(df1) == 2 and len(calls) == 2 and cache_file.exists()

    calls.clear()
    df2 = label_rows(texts, cache_path=cache_file, client_fn=_ok_client(calls))
    assert len(df2) == 2 and len(calls) == 0

    calls.clear()
    df3 = label_rows(texts + ["Message three"], cache_path=cache_file, client_fn=_ok_client(calls))
    assert len(df3) == 3 and len(calls) == 1

    rec = json.loads(cache_file.read_text(encoding="utf-8").splitlines()[0])
    assert set(rec) >= {"key", "text", "response", "model", "ts", "prompt_version"}
    assert len(rec["key"]) == 64  # SHA-256 hex


def test_label_rows_retries_invalid_json_then_succeeds(tmp_path: Path):
    attempts = {"n": 0}

    def flaky(prompt):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return {"content": "oops not json"}
        return {"content": json.dumps(GOOD)}

    df = label_rows(["a"], cache_path=tmp_path / "c.jsonl", client_fn=flaky, sleep_fn=lambda s: None)
    assert len(df) == 1 and attempts["n"] == 3


def test_label_rows_gives_up_without_caching_bad_rows(tmp_path: Path):
    cache = tmp_path / "c.jsonl"
    df = label_rows(["a"], cache_path=cache, client_fn=lambda p: {"content": "nope"}, sleep_fn=lambda s: None)
    assert len(df) == 0
    assert df.attrs["stats"]["failed"] == 1
    assert not cache.exists() or cache.read_text(encoding="utf-8").strip() == ""


def test_label_rows_stops_cleanly_on_daily_quota(tmp_path: Path):
    cache = tmp_path / "c.jsonl"
    state = {"n": 0}

    def client(prompt):
        state["n"] += 1
        if state["n"] > 2:
            raise _RateLimited(daily=True)
        return {"content": json.dumps(GOOD)}

    df = label_rows(["a", "b", "c", "d"], cache_path=cache, client_fn=client, sleep_fn=lambda s: None)
    assert len(df) == 2
    assert df.attrs["stats"]["stopped"] == "quota"
    assert len(cache.read_text(encoding="utf-8").strip().splitlines()) == 2  # progress saved


def test_label_rows_respects_max_calls_and_max_cost(tmp_path: Path):
    df = label_rows(["a", "b", "c"], cache_path=tmp_path / "a.jsonl", client_fn=_ok_client(), max_calls=2)
    assert len(df) == 2 and df.attrs["stats"]["stopped"] == "max_calls"
    df = label_rows(["a", "b", "c"], cache_path=tmp_path / "b.jsonl", client_fn=_ok_client(), max_cost=0.0)
    assert len(df) == 0 and df.attrs["stats"]["stopped"] == "max_cost"


def test_missing_key_raises_clear_error_but_cache_still_works(tmp_path: Path, monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cache = tmp_path / "c.jsonl"
    label_rows(["a"], cache_path=cache, client_fn=_ok_client())  # seed cache with the mock
    # Cached text: no key needed.
    assert len(label_rows(["a"], cache_path=cache)) == 1
    # Uncached text without a key: clear error.
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        label_rows(["brand new text"], cache_path=cache)


def test_redact_removes_key_everywhere():
    secret = "AIzaSECRET123"
    msg = f"Client error for url 'https://x/y?key={secret}' token {secret}"
    out = _redact(msg, secret)
    assert secret not in out and "key=***" in out


# ── filtering ─────────────────────────────────────────────────────────────────

def test_filter_labels():
    raw_df = pd.DataFrame([
        {"key": "k1", "text": "t1", "label": 1, "techniques": ["T2"], "action": 1, "confidence": 0.95},
        {"key": "k2", "text": "t2", "label": 0, "techniques": [], "action": 0, "confidence": 0.98},
        {"key": "k3", "text": "t3", "label": 5, "techniques": [], "action": 0, "confidence": 0.88},
        {"key": "k4", "text": "t4", "label": 2, "techniques": ["T4"], "action": None, "confidence": 0.90},
        {"key": "k5", "text": "t5", "label": 1, "techniques": ["T2"], "action": 1, "confidence": 0.65},
        {"key": "k6", "text": "t6", "label": 0, "techniques": ["T1"], "action": 0, "confidence": 0.92},
        {"key": "k7", "text": "t7", "label": 1, "techniques": ["T2"], "action": 0, "confidence": 0.92},
        {"key": "k8", "text": "t8", "label": 3, "techniques": [], "action": 4, "confidence": 0.92},
        {"key": "k1", "text": "t1", "label": 1, "techniques": ["T2"], "action": 1, "confidence": 0.95},
    ])
    clean_df, reasons = filter_labels(raw_df)
    assert len(clean_df) == 3
    assert reasons["invalid_schema"] == 1
    assert reasons["low_confidence"] == 1
    assert reasons["inconsistent_rules"] == 3
    assert reasons["duplicate"] == 1


def test_filter_labels_rejects_bad_technique_codes():
    df = pd.DataFrame([{"key": "k", "text": "t", "label": 1, "techniques": ["T9"], "action": 1, "confidence": 0.9}])
    clean, reasons = filter_labels(df)
    assert clean.empty and reasons["invalid_schema"] == 1


# ── calibration gate ──────────────────────────────────────────────────────────

def _calib_df(tag: str):
    return pd.DataFrame({
        "id": [101, 102, 103],
        "text": [f"Real OTP {tag} msg", f"Fake prize {tag} msg", f"bKash PIN {tag} claim"],
        "label": [0, 2, 1],
        "techniques": ["NONE", "T4", "T2|T5"],
        "action": [0, 3, 1],
    })


def test_calibrate_prompt_on_real_passing(tmp_path: Path):
    def perfect_client(prompt):
        text = prompt["user"]
        if "Real OTP" in text:
            pred = {"label": 0, "techniques": [], "action": 0, "confidence": 0.95, "language": "english"}
        elif "Fake prize" in text:
            pred = {"label": 2, "techniques": ["T4"], "action": 3, "confidence": 0.90, "language": "english"}
        else:
            pred = {"label": 1, "techniques": ["T2", "T5"], "action": 1, "confidence": 0.92, "language": "banglish"}
        return {"content": json.dumps(pred)}

    report = tmp_path / "report.json"
    res = calibrate_prompt_on_real(
        sample_df=_calib_df("pass"), client_fn=perfect_client, min_macro_f1=0.80,
        cache_path=tmp_path / "calib_pass.jsonl", report_path=report,
    )
    assert res["F_A"] >= 0.80 and "confusion_matrix" in res
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["passed"] is True and saved["prompt_version"] == prompt_version()
    assert _gate_passed(report)[0] is True


def test_calibrate_prompt_on_real_failing(tmp_path: Path):
    def bad_client(prompt):
        return {"content": json.dumps({"label": 0, "techniques": [], "action": 0, "confidence": 0.95, "language": "english"})}

    report = tmp_path / "report_fail.json"
    with pytest.raises(RuntimeError, match="CALIBRATION GATE FAILED"):
        calibrate_prompt_on_real(
            sample_df=_calib_df("fail"), client_fn=bad_client, min_macro_f1=0.80,
            cache_path=tmp_path / "calib_fail.jsonl", report_path=report,
        )
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is False
    assert _gate_passed(report)[0] is False


def test_calibration_never_touches_main_cache(tmp_path: Path):
    main_cache = tmp_path / "llm_labels_v1.jsonl"
    calib_cache = tmp_path / "calib.jsonl"
    calibrate_prompt_on_real(
        sample_df=_calib_df("iso"), min_macro_f1=0.0,
        client_fn=lambda p: {"content": json.dumps({"label": 0, "techniques": [], "action": 0, "confidence": 0.9, "language": "english"})},
        cache_path=calib_cache,
    )
    assert calib_cache.exists() and not main_cache.exists()


def test_gate_requires_same_prompt_and_model(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "model-a")
    report = tmp_path / "r.json"
    base = {"passed": True, "F_A": 0.9, "prompt_version": prompt_version(), "model": "model-a"}
    report.write_text(json.dumps(base), encoding="utf-8")
    assert _gate_passed(report)[0] is True
    report.write_text(json.dumps({**base, "prompt_version": "stale00000"}), encoding="utf-8")
    assert _gate_passed(report)[0] is False
    report.write_text(json.dumps({**base, "model": "model-b"}), encoding="utf-8")
    assert _gate_passed(report)[0] is False
    assert _gate_passed(tmp_path / "missing.json")[0] is False


def test_cli_refuses_label_without_gate(monkeypatch, capsys):
    monkeypatch.setattr(llm_label, "_gate_passed", lambda *a, **k: (False, "no report"))
    rc = llm_label.main(["--label", "--n", "5"])
    assert rc == 2
    assert "REFUSING" in capsys.readouterr().out


# ── provider client (HTTP mocked at the transport layer) ──────────────────────

import httpx as _httpx

_REAL_HTTPX_CLIENT = _httpx.Client  # captured once so repeated patching never nests


def _mock_http(monkeypatch, handler):
    monkeypatch.setattr(llm_label.httpx, "Client",
                        lambda **kw: _REAL_HTTPX_CLIENT(transport=_httpx.MockTransport(handler), **kw))


def test_gemini_client_sends_key_in_header_not_url(monkeypatch):
    import httpx
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key_header"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": json.dumps(GOOD)}]}}],
            "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 40},
        })

    _mock_http(monkeypatch, handler)
    client = llm_label._build_live_client("gemini", "gemini-test-model", "SECRETKEY")
    out = client(build_prompt("hello"))
    assert "SECRETKEY" not in seen["url"] and "key=" not in seen["url"]
    assert seen["key_header"] == "SECRETKEY"
    assert seen["body"]["generationConfig"]["temperature"] == 0.0
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "systemInstruction" in seen["body"]
    assert out["prompt_tokens"] == 900 and json.loads(out["content"])["label"] == 1


def test_gemini_429_daily_vs_per_minute(monkeypatch):
    import httpx

    def daily(request):
        return httpx.Response(429, text='{"error":{"message":"quota","details":[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}')

    def per_minute(request):
        return httpx.Response(429, text='{"error":{"details":[{"quotaId":"GenerateRequestsPerMinutePerProjectPerModel-FreeTier"},{"retryDelay":"34s"}]}}'.replace('{"retryDelay"', '{"@type":"x","retryDelay"'))

    _mock_http(monkeypatch, daily)
    with pytest.raises(_RateLimited) as e1:
        llm_label._build_live_client("gemini", "m", "k")(build_prompt("x"))
    assert e1.value.daily is True

    _mock_http(monkeypatch, per_minute)
    with pytest.raises(_RateLimited) as e2:
        llm_label._build_live_client("gemini", "m", "k")(build_prompt("x"))
    assert e2.value.daily is False and e2.value.retry_after == 34.0


def test_gemini_blocked_response_is_retried_not_crashed(monkeypatch, tmp_path: Path):
    import httpx
    _mock_http(monkeypatch, lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_API_KEY", "k")
    df = label_rows(["x"], cache_path=tmp_path / "c.jsonl", rpm=6000, sleep_fn=lambda s: None)
    assert len(df) == 0 and df.attrs["stats"]["failed"] == 1
