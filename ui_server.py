#!/usr/bin/env python3
"""
Live testing API for gemma_harness.

  python3 ui_server.py
  # serves API on http://127.0.0.1:3847 and static UI from web/dist if built
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from demo.tools import add, flaky_db, get_weather, slow_lookup
from gemma_harness import Client

ROOT = Path(__file__).resolve().parent
MODEL = "gemma-4-26b-a4b-it"
TOOLS = [get_weather, add, slow_lookup, flaky_db]
PORT = 3847


def load_dotenv() -> None:
    env_path = ROOT / ".env"
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

app = FastAPI(title="Gemma Harness Live Lab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    mode: Literal["harness", "plain", "compare"] = "harness"
    tool_timeout_s: float = Field(default=3.0, ge=0.5, le=30.0)
    max_steps: int = Field(default=8, ge=1, le=16)


SAMPLE_PROMPTS = [
    {"label": "Phantom tool", "prompt": "Use web_search to find today's news in Karachi."},
    {"label": "Typo tool name", "prompt": "What's the weather in Lahore? Use get_weather_now."},
    {"label": "String args", "prompt": 'Add "7" and "8" with the add tool.'},
    {"label": "Wrong arg name", "prompt": "Get weather for Karachi but call the argument 'location'."},
    {"label": "Tool timeout", "prompt": "Look up the population of Pakistan with slow_lookup."},
    {"label": "Tool crash", "prompt": "Fetch user 42 with flaky_db and tell me their name."},
    {"label": "Dedupe loop", "prompt": "Check weather in Karachi. Check again. Once more."},
    {"label": "Simple math", "prompt": "Hi! What's 2+2?"},
]


@app.get("/api/health")
def health() -> dict[str, Any]:
    key = bool((os.environ.get("GEMINI_API_KEY") or "").strip())
    return {"ok": True, "model": MODEL, "api_key_set": key, "tools": [f.__name__ for f in TOOLS]}


@app.get("/api/samples")
def samples() -> list[dict[str, str]]:
    return SAMPLE_PROMPTS


def _run_harness(prompt: str, tool_timeout_s: float, max_steps: int, emit: Callable[[dict], None]) -> dict:
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip() or None
    events: list[dict] = []

    def on_event(ev: dict) -> None:
        events.append(ev)
        emit({"type": "event", "mode": "harness", "event": ev})

    h = Client(
        model=MODEL,
        api_key=api_key,
        max_steps=max_steps,
        tool_timeout_s=tool_timeout_s,
        route_thinking=True,
        system_instruction="You are a helpful assistant.",
        on_event=on_event,
    )
    t0 = time.time()
    out = h.run(prompt, tools=TOOLS)
    latency = round(time.time() - t0, 3)
    result = {
        "mode": "harness",
        "text": out.get("text") or "",
        "steps": out.get("steps") or 0,
        "latency_s": latency,
        "recoveries": [e["label"] for e in events if e.get("label") not in {"CALL", "FINAL", "TOOL_OK"}],
        "events": events,
        "crashed": False,
        "error": None,
    }
    emit({"type": "result", **result})
    return result


def _run_plain(prompt: str, tool_timeout_s: float, max_steps: int, emit: Callable[[dict], None]) -> dict:
    """Minimal plain loop for side-by-side compare (no recovery)."""
    import concurrent.futures as cf
    import random

    from google import genai
    from google.genai import types, errors

    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip() or None
    c = genai.Client(api_key=api_key)
    tm = {f.__name__: f for f in TOOLS}
    decls = [types.FunctionDeclaration.from_callable(client=c, callable=f) for f in TOOLS]
    cfg = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=decls)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    t0 = time.time()
    text_out = ""
    steps = 0
    try:
        for step in range(1, max_steps + 1):
            steps = step
            emit({"type": "event", "mode": "plain", "event": {"label": "CALL", "step": step}})
            r = None
            for attempt in range(5):
                try:
                    r = c.models.generate_content(model=MODEL, contents=contents, config=cfg)
                    break
                except errors.APIError as e:
                    if e.code in (429, 500, 503) and attempt < 4:
                        emit({"type": "event", "mode": "plain", "event": {
                            "label": "BACKOFF_429" if e.code == 429 else "BACKOFF_5XX",
                            "step": step,
                        }})
                        time.sleep(min(30, 2 ** attempt) + random.random())
                        continue
                    raise
            assert r is not None
            calls = r.function_calls or []
            if not calls:
                text_out = (r.text or "").strip()
                emit({"type": "event", "mode": "plain", "event": {
                    "label": "FINAL", "step": step, "chars": len(text_out),
                }})
                break
            contents.append(r.candidates[0].content)
            parts = []
            for fc in calls:
                name = fc.name or ""
                args = dict(fc.args or {})
                if name not in tm:
                    raise KeyError(f"unknown tool: {name}")
                with cf.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(tm[name], **args)
                    result = fut.result(timeout=tool_timeout_s)
                emit({"type": "event", "mode": "plain", "event": {
                    "label": "TOOL_OK", "step": step, "tool": name,
                }})
                parts.append(types.Part.from_function_response(name=name, response={"result": result}))
            contents.append(types.Content(role="user", parts=parts))
        latency = round(time.time() - t0, 3)
        result = {
            "mode": "plain",
            "text": text_out,
            "steps": steps,
            "latency_s": latency,
            "recoveries": [],
            "events": [],
            "crashed": False,
            "error": None,
        }
        emit({"type": "result", **result})
        return result
    except Exception as e:
        latency = round(time.time() - t0, 3)
        result = {
            "mode": "plain",
            "text": "",
            "steps": steps,
            "latency_s": latency,
            "recoveries": [],
            "events": [],
            "crashed": True,
            "error": f"{type(e).__name__}: {e}",
        }
        emit({"type": "event", "mode": "plain", "event": {
            "label": "CRASH", "err": result["error"][:200],
        }})
        emit({"type": "result", **result})
        return result


@app.post("/api/run")
def run_sync(req: RunRequest) -> dict[str, Any]:
    if not (os.environ.get("GEMINI_API_KEY") or "").strip():
        raise HTTPException(400, "GEMINI_API_KEY is not set. Add it to .env")
    collected: list[dict] = []

    def emit(msg: dict) -> None:
        collected.append(msg)

    if req.mode == "harness":
        return _run_harness(req.prompt, req.tool_timeout_s, req.max_steps, emit)
    if req.mode == "plain":
        return _run_plain(req.prompt, req.tool_timeout_s, req.max_steps, emit)
    # compare
    h = _run_harness(req.prompt, req.tool_timeout_s, req.max_steps, emit)
    p = _run_plain(req.prompt, req.tool_timeout_s, req.max_steps, emit)
    return {"mode": "compare", "harness": h, "plain": p}


@app.post("/api/run/stream")
def run_stream(req: RunRequest):
    if not (os.environ.get("GEMINI_API_KEY") or "").strip():
        raise HTTPException(400, "GEMINI_API_KEY is not set. Add it to .env")

    q: queue.Queue[dict | None] = queue.Queue()

    def emit(msg: dict) -> None:
        q.put(msg)

    def worker() -> None:
        try:
            emit({"type": "start", "mode": req.mode, "model": MODEL, "prompt": req.prompt})
            if req.mode == "compare":
                _run_harness(req.prompt, req.tool_timeout_s, req.max_steps, emit)
                _run_plain(req.prompt, req.tool_timeout_s, req.max_steps, emit)
            elif req.mode == "plain":
                _run_plain(req.prompt, req.tool_timeout_s, req.max_steps, emit)
            else:
                _run_harness(req.prompt, req.tool_timeout_s, req.max_steps, emit)
            emit({"type": "done"})
        except Exception as e:
            emit({"type": "error", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]})
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


dist = ROOT / "web" / "dist"
if dist.is_dir():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(dist / "index.html")

    @app.get("/{path:path}")
    def spa(path: str):
        candidate = dist / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")


if __name__ == "__main__":
    import uvicorn

    print(f"Gemma Harness Live Lab → http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
