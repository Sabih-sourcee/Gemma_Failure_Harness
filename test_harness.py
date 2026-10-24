"""
Smoke test for gemma_harness recovery events.

Run from the project root:
  python3 test_harness.py

Without GEMINI_API_KEY, uses a mocked model response that injects the three
failure modes the harness is meant to catch. With GEMINI_API_KEY set, also
runs a short live prompt against Gemma 4.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from google.genai import types

from demo.tools import add, flaky_db, get_weather
from gemma_harness import Client


REQUIRED_LABELS = ("TOOL_NAME_FIXED", "ARGS_COERCED", "TOOL_ERROR")


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ (does not override existing)."""
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


load_dotenv()


def on_event(ev: dict) -> None:
    detail = {k: v for k, v in ev.items() if k not in ("t", "step", "label")}
    arrow = f" → {detail}" if detail else ""
    print(f"EVENT: {ev['label']}{arrow}")


def _tool_call_response(*calls: tuple[str, dict]) -> types.GenerateContentResponse:
    parts = [types.Part.from_function_call(name=n, args=a) for n, a in calls]
    content = types.Content(role="model", parts=parts)
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=content, finish_reason="STOP")]
    )


def _text_response(text: str) -> types.GenerateContentResponse:
    content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=content, finish_reason="STOP")]
    )


def run_mocked() -> dict:
    """Inject typo name, string numbers, and a crashing tool — no API needed."""
    print("=== Mocked recovery smoke test ===")
    h = Client(
        api_key="mock-key",
        model="gemma-4-26b-a4b-it",
        max_steps=4,
        route_thinking=False,
        system_instruction="You are a helpful assistant.",
        on_event=on_event,
    )

    queue = [
        # Typo tool name + misspelled arg, stringy numbers, crashing tool
        _tool_call_response(
            ("get_wether", {"citty": "Karachi"}),
            ("add", {"a": "10", "b": "20"}),
            ("flaky_db", {"user_id": 42}),
        ),
        _text_response(
            "Karachi is 31C. 10+20=30. The user DB was down, so no name."
        ),
    ]

    def fake_generate_content(**kwargs):
        if not queue:
            return _text_response("fallback")
        return queue.pop(0)

    h.raw.models.generate_content = fake_generate_content  # type: ignore[method-assign]

    out = h.run(
        "Weather in Karachi (get_wether), add 10 and 20, then fetch user 42.",
        tools=[get_weather, add, flaky_db],
    )
    print("\nFinal answer:", out["text"])
    print("Steps:", out["steps"])
    return out


def run_live() -> dict | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("\n=== Live Gemma run skipped (GEMINI_API_KEY not set) ===")
        return None

    print("\n=== Live Gemma recovery smoke test ===")
    h = Client(
        model="gemma-4-26b-a4b-it",
        max_steps=6,
        system_instruction="You are a helpful assistant. Prefer tools when available.",
        on_event=on_event,
    )
    out = h.run(
        "What's the weather in Lahore? Use get_weather_now. "
        "Then add \"7\" and \"8\" with the add tool. "
        "Then fetch user 42 with flaky_db and tell me their name.",
        tools=[get_weather, add, flaky_db],
    )
    print("\nFinal answer:", out["text"])
    print("Steps:", out["steps"])
    return out


def assert_required(trace: list[dict], source: str) -> bool:
    labels = {ev["label"] for ev in trace}
    missing = [lab for lab in REQUIRED_LABELS if lab not in labels]
    if missing:
        print(f"\nFAIL ({source}): missing labels: {missing}")
        print("Saw:", sorted(labels))
        return False
    print(f"\nPASS ({source}): saw TOOL_NAME_FIXED, ARGS_COERCED, TOOL_ERROR")
    return True


def main() -> int:
    mocked = run_mocked()
    ok = assert_required(mocked["trace"], "mocked")

    live = run_live()
    if live is not None:
        # Live model may not misspell every time; report what it did emit.
        live_labels = {ev["label"] for ev in live["trace"]}
        print("Live labels:", sorted(live_labels))
        for lab in REQUIRED_LABELS:
            status = "yes" if lab in live_labels else "not this run"
            print(f"  {lab}: {status}")

    if ok:
        print("\nConfirmed")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
