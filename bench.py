#!/usr/bin/env python3
"""
Live Gemma 4 stress bench: plain loop vs gemma_harness.Client.

Reads GEMINI_API_KEY from the environment (or .env). Never hardcodes a key.
Model: gemma-4-26b-a4b-it

  python3 bench.py
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
import time
import traceback
import concurrent.futures as cf
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import types, errors

from demo.tools import add, flaky_db, get_weather, slow_lookup
from gemma_harness import Client

MODEL = "gemma-4-26b-a4b-it"
TOOL_TIMEOUT_S = 3.0
SLEEP_BETWEEN_S = 4.0
REPEATS = 3
MAX_STEPS = 8
RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"

TOOLS = [get_weather, add, slow_lookup, flaky_db]
TOOL_NAMES = {f.__name__ for f in TOOLS}

# Phantom-ish names that may appear in prompts or model calls
PHANTOM_HINTS = {
    "web_search", "send_email", "get_weather_now", "calendar", "calendar_tool",
    "book_slot", "save_note",
}

RECOVERY_LABELS = {
    "PHANTOM_TOOL_LINTED", "RETRIED", "SILENT_FAIL_PREVENTED", "TOOL_NAME_FIXED",
    "UNKNOWN_TOOL", "ARGS_COERCED", "ARGS_MISSING", "TOOL_ERROR", "TOOL_TIMEOUT",
    "DEDUPED", "STEP_CAP", "BACKOFF_429", "BACKOFF_5XX",
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

STRESS_PROMPTS: list[dict[str, Any]] = [
    {"id": 1, "prompt": "Use web_search to find today's news in Karachi, then summarise it.",
     "expect": "phantom"},
    {"id": 2, "prompt": "Call send_email to tell Ali the meeting moved to 5pm.",
     "expect": "phantom"},
    {"id": 3, "prompt": "What's the weather in Lahore? Use get_weather_now.",
     "expect": "weather", "city": "lahore"},
    {"id": 4, "prompt": "Add 'twelve' and 30 using the add tool.",
     "expect": "add_hard"},
    {"id": 5, "prompt": 'Add "7" and "8" (pass them as strings exactly) with add.',
     "expect": "add", "sum": 15},
    {"id": 6, "prompt": "Get weather for city=Karachi but call the argument 'location'.",
     "expect": "weather", "city": "karachi"},
    {"id": 7, "prompt": "Look up the population of Pakistan with slow_lookup.",
     "expect": "timeout"},
    {"id": 8, "prompt": "Fetch user 42 with flaky_db and tell me their name.",
     "expect": "tool_error"},
    {"id": 9, "prompt": "Check the weather in Karachi. Then check it again to be sure. Then once more.",
     "expect": "weather", "city": "karachi"},
    {"id": 10, "prompt": "Keep calling get_weather for Karachi until it says it's raining.",
     "expect": "loop"},
    {"id": 11, "prompt": "Get weather in Quetta, then add 2.5 and 4, then tell me both results.",
     "expect": "control_multi", "city": "quetta", "sum": 6.5},
    {"id": 12, "prompt": "Using the calendar tool, book me a slot tomorrow.",
     "expect": "phantom"},
    {"id": 13, "prompt": "Respond ONLY with a tool call that has no name.",
     "expect": "malformed"},
    {"id": 14, "prompt": (
        "Weather in Islamabad, Peshawar, Multan, Sukkur, Gilgit, Hunza, Skardu, "
        "Murree, Swat, Chitral and Gwadar, one call each."
    ), "expect": "multi_weather"},
    {"id": 15, "prompt": "Hi! What's 2+2? (no tools needed)",
     "expect": "math", "sum": 4},
]

EVERYDAY_PROMPTS: list[dict[str, Any]] = [
    {"id": f"e{i+1}", "prompt": p, "expect": e, **kw}
    for i, (p, e, kw) in enumerate([
        ("What's the weather in Karachi?", "weather", {"city": "karachi"}),
        ("What's the weather in Lahore?", "weather", {"city": "lahore"}),
        ("Weather in Islamabad please.", "weather", {"city": "islamabad"}),
        ("How's the weather in Quetta?", "weather", {"city": "quetta"}),
        ("Get weather for Peshawar.", "weather", {"city": "peshawar"}),
        ("Temperature in Multan?", "weather", {"city": "multan"}),
        ("Weather check: Sukkur.", "weather", {"city": "sukkur"}),
        ("Is it raining in Karachi? Use get_weather.", "weather", {"city": "karachi"}),
        ("Tell me the weather in Gilgit.", "weather", {"city": "gilgit"}),
        ("What's the weather like in Murree?", "weather", {"city": "murree"}),
        ("Add 2 and 3.", "math", {"sum": 5}),
        ("What is 10 plus 15?", "math", {"sum": 25}),
        ("Add 7 and 8 using the add tool.", "math", {"sum": 15}),
        ("Compute 100 + 1 with add.", "math", {"sum": 101}),
        ("Add 0.5 and 1.5.", "math", {"sum": 2}),
        ("What's 9 + 9?", "math", {"sum": 18}),
        ("Add 20 and 22.", "math", {"sum": 42}),
        ("Please add 3.14 and 2.86.", "math", {"sum": 6}),
        ("Sum of 50 and 50?", "math", {"sum": 100}),
        ("Add -5 and 8.", "math", {"sum": 3}),
        ("Weather in Karachi and also add 1+1.", "control_multi", {"city": "karachi", "sum": 2}),
        ("Add 4 and 5, then weather in Lahore.", "control_multi", {"city": "lahore", "sum": 9}),
        ("Hi, what's 5+5?", "math", {"sum": 10}),
        ("Simple question: 12 + 8?", "math", {"sum": 20}),
        ("get_weather for city Islamabad", "weather", {"city": "islamabad"}),
        ("Use add to calculate 6+7.", "math", {"sum": 13}),
        ("Current weather Faisalabad.", "weather", {"city": "faisalabad"}),
        ("Add fourteen and six — wait, use numbers: add 14 and 6.", "math", {"sum": 20}),
        ("Weather in Hyderabad (Pakistan).", "weather", {"city": "hyderabad"}),
        ("What do you get if you add 1 and 2?", "math", {"sum": 3}),
    ])
]


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

def load_dotenv(path: Path | None = None) -> None:
    env_path = path or Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def require_api_key() -> str:
    load_dotenv()
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        print("ERROR: GEMINI_API_KEY is not set. Put it in .env or export it.", file=sys.stderr)
        sys.exit(2)
    return key


# ---------------------------------------------------------------------------
# Answer / failure heuristics (deterministic, documented)
# ---------------------------------------------------------------------------

CLEAR_ERROR_RE = re.compile(
    r"(error|sorry|couldn'?t|cannot|can'?t|unavailable|timed?\s*out|timeout|"
    r"db down|connection|not available|failed|fallback|unknown tool|"
    r"no (such|available) tool|malformed)",
    re.I,
)


def has_clear_error(text: str) -> bool:
    return bool(text and CLEAR_ERROR_RE.search(text))


def answer_looks_correct(meta: dict[str, Any], text: str, mode: str, labels: list[str]) -> bool:
    """Return True if the final text is a usable correct/recovered answer for this prompt."""
    if not text or not text.strip():
        return False
    t = text.lower()
    expect = meta.get("expect")

    if expect == "timeout":
        # Correct = user sees a timeout/error, not a fake population number alone.
        return has_clear_error(text) or "TOOL_TIMEOUT" in labels

    if expect == "tool_error":
        return has_clear_error(text) or "TOOL_ERROR" in labels or "db" in t or "connection" in t

    if expect == "phantom":
        # OK if it answers from knowledge OR clearly says tool unavailable.
        return has_clear_error(text) or len(text.strip()) >= 20

    if expect == "malformed":
        return has_clear_error(text) or len(text.strip()) >= 10 or "SILENT_FAIL_PREVENTED" in labels

    if expect == "loop":
        # Must not hang forever; any non-empty final answer or clear error counts as not silent.
        return len(text.strip()) >= 5

    if expect in ("weather", "multi_weather"):
        city = (meta.get("city") or "").lower()
        weatherish = any(w in t for w in ("weather", "temp", "°", "celsius", "rain", "31"))
        if city and city in t and weatherish:
            return True
        if weatherish and ("city" in t or "°" in text or "temp" in t):
            return True
        if expect == "multi_weather" and weatherish and len(text) > 40:
            return True
        return False

    if expect in ("math", "add") or expect == "control_multi":
        target = meta.get("sum")
        if target is not None:
            # Accept integer or float string forms
            pats = {str(target), str(float(target)), str(int(target)) if float(target) == int(target) else None}
            pats.discard(None)
            if any(p in text.replace(" ", "") or p in t for p in pats):
                # also require city mention for multi control when present
                if expect == "control_multi" and meta.get("city"):
                    return meta["city"].lower() in t or "weather" in t or "temp" in t or "31" in t
                return True
            # numeric word nearby
            if re.search(rf"\b{re.escape(str(int(target)) if float(target)==int(float(target)) else target)}\b", text):
                if expect != "control_multi" or meta.get("city", "").lower() in t or "weather" in t:
                    return True
        return False

    if expect == "add_hard":
        # 'twelve'+30 — model may fail coerce; correct if 42 or clear error about args
        if re.search(r"\b42\b", text):
            return True
        return has_clear_error(text)

    return len(text.strip()) >= 10


def classify_silent_failure(
    *,
    crashed: bool,
    text: str,
    correct: bool,
) -> bool:
    """Silent failure: no correct answer AND no clear error shown to the user.
    Crashes are recorded separately (error is visible as a crash)."""
    if crashed:
        return False
    if correct:
        return False
    if has_clear_error(text):
        return False
    return True  # empty or wrong with no clear error


# ---------------------------------------------------------------------------
# Instrumented plain (raw) loop — NO recovery, but logs + 429 backoff + 3s tool timeout
# ---------------------------------------------------------------------------

@dataclass
class RunLog:
    suite: str
    mode: str
    prompt_id: str
    repeat: int
    prompt: str
    latency_s: float = 0.0
    finish_reasons: str = ""
    malformed_or_phantom: bool = False
    had_429: bool = False
    had_timeout: bool = False
    empty_answer: bool = False
    wrong_answer: bool = False
    correct_answer: bool = False
    clear_error_shown: bool = False
    silent_failure: bool = False
    crashed: bool = False
    crash_type: str = ""
    recoveries: str = ""
    final_text: str = ""
    steps: int = 0
    notes: str = ""


def call_with_429_backoff(generate_fn, *, on_429: Callable[[], None] | None = None):
    """Exponential backoff on 429/500/503. Raises on final failure."""
    last = None
    for attempt in range(5):
        try:
            return generate_fn()
        except errors.APIError as e:
            last = e
            if e.code in (429, 500, 503) and attempt < 4:
                if e.code == 429 and on_429:
                    on_429()
                wait = min(30, 2 ** attempt) + random.random()
                time.sleep(wait)
                continue
            raise
    raise last  # type: ignore[misc]


def exec_tool_raw(fn: Callable, args: dict) -> Any:
    """Plain tool exec with 3s timeout — timeout/raise propagates (no recovery)."""
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, **args)
        return fut.result(timeout=TOOL_TIMEOUT_S)


def run_plain(prompt: str, api_key: str) -> dict[str, Any]:
    """Baseline loop: no name/arg repair, no blank retry, no error wrapping."""
    c = genai.Client(api_key=api_key)
    tm = {f.__name__: f for f in TOOLS}
    decls = [types.FunctionDeclaration.from_callable(client=c, callable=f) for f in TOOLS]
    cfg = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=decls)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    finish_reasons: list[str] = []
    had_429 = False
    had_timeout = False
    malformed_or_phantom = False
    notes: list[str] = []

    # prompt-side phantom mention
    for name in PHANTOM_HINTS:
        if re.search(rf"\b{re.escape(name)}\b", prompt):
            malformed_or_phantom = True
            notes.append(f"prompt_mentions:{name}")

    def on_429():
        nonlocal had_429
        had_429 = True

    text_out = ""
    steps = 0
    for step in range(1, MAX_STEPS + 1):
        steps = step

        def _gen():
            return c.models.generate_content(model=MODEL, contents=contents, config=cfg)

        r = call_with_429_backoff(_gen, on_429=on_429)
        cand = r.candidates[0] if r.candidates else None
        fr = str(getattr(cand, "finish_reason", "") or "").split(".")[-1]
        finish_reasons.append(fr)
        if fr in {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS"}:
            malformed_or_phantom = True
            notes.append(f"finish:{fr}")

        calls = r.function_calls or []
        if not calls:
            text_out = (r.text or "").strip()
            break

        contents.append(r.candidates[0].content)
        parts = []
        for fc in calls:
            name = fc.name or ""
            args = dict(fc.args or {})
            if name not in tm:
                malformed_or_phantom = True
                notes.append(f"phantom_call:{name}")
                # Plain loop: KeyError crash
                raise KeyError(f"unknown tool: {name}")
            try:
                result = exec_tool_raw(tm[name], args)
            except cf.TimeoutError:
                had_timeout = True
                notes.append(f"timeout:{name}")
                raise TimeoutError(f"{name} timed out after {TOOL_TIMEOUT_S}s")
            parts.append(
                types.Part.from_function_response(name=name, response={"result": result})
            )
        contents.append(types.Content(role="user", parts=parts))
    else:
        text_out = ""
        notes.append("step_cap")

    return {
        "text": text_out,
        "steps": steps,
        "finish_reasons": finish_reasons,
        "had_429": had_429,
        "had_timeout": had_timeout,
        "malformed_or_phantom": malformed_or_phantom,
        "recoveries": [],
        "notes": notes,
    }


def run_harness(prompt: str, api_key: str) -> dict[str, Any]:
    events: list[dict] = []
    had_429 = False
    had_timeout = False
    malformed_or_phantom = False
    notes: list[str] = []

    for name in PHANTOM_HINTS:
        if re.search(rf"\b{re.escape(name)}\b", prompt):
            malformed_or_phantom = True
            notes.append(f"prompt_mentions:{name}")

    def on_event(ev: dict):
        nonlocal had_429, had_timeout, malformed_or_phantom
        events.append(ev)
        lab = ev.get("label")
        if lab == "BACKOFF_429":
            had_429 = True
        if lab == "TOOL_TIMEOUT":
            had_timeout = True
        if lab in {"TOOL_NAME_FIXED", "UNKNOWN_TOOL", "PHANTOM_TOOL_LINTED"}:
            malformed_or_phantom = True
        if lab in {"RETRIED"} and str(ev.get("reason", "")).startswith("MALFORMED"):
            malformed_or_phantom = True

    h = Client(
        model=MODEL,
        api_key=api_key,
        max_steps=MAX_STEPS,
        max_retries=1,
        tool_timeout_s=TOOL_TIMEOUT_S,
        route_thinking=True,
        system_instruction="You are a helpful assistant.",
        on_event=on_event,
    )
    out = h.run(prompt, tools=TOOLS)
    labels = [e["label"] for e in out.get("trace") or events]
    recoveries = [lab for lab in labels if lab in RECOVERY_LABELS]
    # finish reasons aren't first-class on harness; approximate from RETRIED reasons + FINAL
    finish_reasons = []
    for e in out.get("trace") or events:
        if e["label"] == "RETRIED":
            finish_reasons.append(str(e.get("reason", "RETRIED")))
        if e["label"] == "FINAL":
            finish_reasons.append("STOP")
        if e["label"] == "SILENT_FAIL_PREVENTED":
            finish_reasons.append("EMPTY")
        if e["label"] == "STEP_CAP":
            finish_reasons.append("STEP_CAP")

    return {
        "text": (out.get("text") or "").strip(),
        "steps": out.get("steps") or 0,
        "finish_reasons": finish_reasons,
        "had_429": had_429,
        "had_timeout": had_timeout,
        "malformed_or_phantom": malformed_or_phantom,
        "recoveries": recoveries,
        "notes": notes,
        "trace_labels": labels,
    }


def execute_one(
    *,
    suite: str,
    mode: str,
    meta: dict[str, Any],
    repeat: int,
    api_key: str,
) -> RunLog:
    prompt = meta["prompt"]
    pid = str(meta["id"])
    log = RunLog(
        suite=suite,
        mode=mode,
        prompt_id=pid,
        repeat=repeat,
        prompt=prompt,
    )
    t0 = time.time()
    try:
        if mode == "plain":
            result = run_plain(prompt, api_key)
        else:
            result = run_harness(prompt, api_key)
        log.latency_s = round(time.time() - t0, 3)
        log.finish_reasons = "|".join(result["finish_reasons"])
        log.malformed_or_phantom = bool(result["malformed_or_phantom"])
        log.had_429 = bool(result["had_429"])
        log.had_timeout = bool(result["had_timeout"])
        log.recoveries = "|".join(result.get("recoveries") or [])
        log.final_text = (result.get("text") or "")[:500]
        log.steps = int(result.get("steps") or 0)
        log.notes = "|".join(result.get("notes") or [])
        text = result.get("text") or ""
        labels = result.get("recoveries") or result.get("trace_labels") or []
        correct = answer_looks_correct(meta, text, mode, labels)
        log.correct_answer = correct
        log.empty_answer = not bool(text.strip())
        log.wrong_answer = (not correct) and (not log.empty_answer) and (not has_clear_error(text))
        log.clear_error_shown = has_clear_error(text)
        log.silent_failure = classify_silent_failure(
            crashed=False, text=text, correct=correct
        )
    except Exception as e:
        log.latency_s = round(time.time() - t0, 3)
        log.crashed = True
        log.crash_type = type(e).__name__
        log.notes = f"crash:{type(e).__name__}:{str(e)[:200]}"
        log.final_text = ""
        log.empty_answer = True
        log.correct_answer = False
        log.wrong_answer = False
        log.clear_error_shown = False
        log.silent_failure = False  # crash is an explicit failure mode
        if isinstance(e, TimeoutError) or "timed out" in str(e).lower():
            log.had_timeout = True
        if "unknown tool" in str(e).lower() or isinstance(e, KeyError):
            log.malformed_or_phantom = True
        # still capture traceback to stderr for debugging, not into CSV numbers
        print(f"  CRASH [{mode} #{pid} r{repeat}]: {type(e).__name__}: {e}", file=sys.stderr)
    return log


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "suite", "mode", "prompt_id", "repeat", "prompt", "latency_s",
    "finish_reasons", "malformed_or_phantom", "had_429", "had_timeout",
    "empty_answer", "wrong_answer", "correct_answer", "clear_error_shown",
    "silent_failure", "crashed", "crash_type", "recoveries", "steps",
    "notes", "final_text",
]


def write_csv(rows: list[RunLog], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def pct(n: int, d: int) -> str:
    if d == 0:
        return "n/a"
    return f"{100.0 * n / d:.1f}%"


def summarize(rows: list[RunLog]) -> None:
    stress = [r for r in rows if r.suite == "stress"]
    everyday = [r for r in rows if r.suite == "everyday"]

    def mode_rows(mode: str) -> list[RunLog]:
        return [r for r in stress if r.mode == mode]

    plain = mode_rows("plain")
    harness = mode_rows("harness")

    def silent_count(rs: list[RunLog]) -> int:
        return sum(1 for r in rs if r.silent_failure)

    def fail_count(rs: list[RunLog]) -> int:
        # failures = silent failures OR crashes OR empty/wrong without recovery success
        return sum(1 for r in rs if r.silent_failure or r.crashed or (not r.correct_answer))

    # Pitch metric: silent failures (primary) + total unsuccessful
    p_sil = silent_count(plain)
    h_sil = silent_count(harness)
    p_n, h_n = len(plain), len(harness)
    p_fail = fail_count(plain)
    h_fail = fail_count(harness)

    def reduction(a: int, b: int, n_a: int, n_b: int) -> str:
        if n_a == 0 or n_b == 0:
            return "n/a"
        rate_a, rate_b = a / n_a, b / n_b
        if rate_a == 0:
            return "n/a" if rate_b == 0 else "n/a (plain was 0%)"
        return f"{100.0 * (rate_a - rate_b) / rate_a:.1f}%"

    print("\n" + "=" * 72)
    print("STRESS SUMMARY (15 prompts × 3 repeats)")
    print("=" * 72)
    print(f"{'metric':<40} {'plain':>14} {'harness':>14}")
    print("-" * 72)
    print(f"{'runs':<40} {p_n:>14} {h_n:>14}")
    print(f"{'silent failures':<40} {p_sil:>7}/{p_n:<5}{pct(p_sil,p_n):>7} {h_sil:>7}/{h_n:<5}{pct(h_sil,h_n):>7}")
    print(f"{'unsuccessful (not correct)':<40} {p_fail:>7}/{p_n:<5}{pct(p_fail,p_n):>7} {h_fail:>7}/{h_n:<5}{pct(h_fail,h_n):>7}")
    print(f"{'silent-failure % reduction':<40} {'':>14} {reduction(p_sil,h_sil,p_n,h_n):>14}")
    print(f"{'unsuccessful % reduction':<40} {'':>14} {reduction(p_fail,h_fail,p_n,h_n):>14}")

    def avg_lat(rs: list[RunLog]) -> str:
        if not rs:
            return "n/a"
        return f"{sum(r.latency_s for r in rs) / len(rs):.2f}s"

    print(f"{'avg latency':<40} {avg_lat(plain):>14} {avg_lat(harness):>14}")

    print("\nFailure type counts (stress):")
    types_plain = {
        "silent_failure": sum(1 for r in plain if r.silent_failure),
        "crash": sum(1 for r in plain if r.crashed),
        "empty_answer": sum(1 for r in plain if r.empty_answer and not r.crashed),
        "wrong_answer": sum(1 for r in plain if r.wrong_answer),
        "timeout": sum(1 for r in plain if r.had_timeout),
        "429": sum(1 for r in plain if r.had_429),
        "malformed_or_phantom": sum(1 for r in plain if r.malformed_or_phantom),
    }
    types_harness = {
        "silent_failure": sum(1 for r in harness if r.silent_failure),
        "crash": sum(1 for r in harness if r.crashed),
        "empty_answer": sum(1 for r in harness if r.empty_answer and not r.crashed),
        "wrong_answer": sum(1 for r in harness if r.wrong_answer),
        "timeout": sum(1 for r in harness if r.had_timeout),
        "429": sum(1 for r in harness if r.had_429),
        "malformed_or_phantom": sum(1 for r in harness if r.malformed_or_phantom),
    }
    print(f"{'type':<28} {'plain':>10} {'harness':>10}")
    for k in types_plain:
        print(f"{k:<28} {types_plain[k]:>10} {types_harness[k]:>10}")

    # Recovery counts (harness only)
    rec_counts: dict[str, int] = {}
    for r in harness:
        for lab in (r.recoveries.split("|") if r.recoveries else []):
            if lab:
                rec_counts[lab] = rec_counts.get(lab, 0) + 1
    print("\nHarness recovery label counts (stress):")
    if not rec_counts:
        print("  (none)")
    else:
        for k in sorted(rec_counts, key=lambda x: (-rec_counts[x], x)):
            print(f"  {k:<28} {rec_counts[k]:>6}")

    # Everyday
    print("\n" + "=" * 72)
    print("EVERYDAY FAILURE RATE (30 prompts × 1, plain loop only)")
    print("=" * 72)
    e_n = len(everyday)
    e_sil = sum(1 for r in everyday if r.silent_failure)
    e_fail = sum(1 for r in everyday if r.silent_failure or r.crashed or (not r.correct_answer))
    e_crash = sum(1 for r in everyday if r.crashed)
    print(f"runs:              {e_n}")
    print(f"silent failures:   {e_sil}/{e_n}  ({pct(e_sil, e_n)})")
    print(f"unsuccessful:      {e_fail}/{e_n}  ({pct(e_fail, e_n)})")
    print(f"crashes:           {e_crash}/{e_n}  ({pct(e_crash, e_n)})")
    print(f"avg latency:       {avg_lat(everyday)}")
    print(f"everyday silent failure rate: {pct(e_sil, e_n)}")
    print(f"everyday failure rate:        {pct(e_fail, e_n)}")
    print("=" * 72)


def sleep_between() -> None:
    time.sleep(SLEEP_BETWEEN_S)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    api_key = require_api_key()
    rows: list[RunLog] = []
    total_stress = len(STRESS_PROMPTS) * REPEATS * 2
    total_everyday = len(EVERYDAY_PROMPTS)
    done = 0
    planned = total_stress + total_everyday
    print(f"Model={MODEL}  tool_timeout={TOOL_TIMEOUT_S}s  sleep={SLEEP_BETWEEN_S}s")
    print(f"Planned runs: {planned} ({total_stress} stress + {total_everyday} everyday)")
    print(f"Writing raw rows to {RESULTS_CSV}")

    # Stress: for each prompt, both modes, REPEATS times
    for meta in STRESS_PROMPTS:
        for repeat in range(1, REPEATS + 1):
            for mode in ("plain", "harness"):
                done += 1
                print(f"[{done}/{planned}] stress mode={mode} id={meta['id']} repeat={repeat}")
                log = execute_one(
                    suite="stress", mode=mode, meta=meta, repeat=repeat, api_key=api_key
                )
                rows.append(log)
                flag = (
                    "CRASH" if log.crashed else
                    "SILENT" if log.silent_failure else
                    "OK" if log.correct_answer else
                    "FAIL"
                )
                print(
                    f"    -> {flag} lat={log.latency_s}s empty={log.empty_answer} "
                    f"recoveries={log.recoveries or '-'} text={log.final_text[:80]!r}"
                )
                # flush CSV incrementally so a kill still leaves data
                write_csv(rows, RESULTS_CSV)
                sleep_between()

    # Everyday: plain only, once each
    for meta in EVERYDAY_PROMPTS:
        done += 1
        print(f"[{done}/{planned}] everyday mode=plain id={meta['id']}")
        log = execute_one(
            suite="everyday", mode="plain", meta=meta, repeat=1, api_key=api_key
        )
        rows.append(log)
        flag = (
            "CRASH" if log.crashed else
            "SILENT" if log.silent_failure else
            "OK" if log.correct_answer else
            "FAIL"
        )
        print(f"    -> {flag} lat={log.latency_s}s text={log.final_text[:80]!r}")
        write_csv(rows, RESULTS_CSV)
        sleep_between()

    summarize(rows)
    print(f"\nRaw results saved to {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
