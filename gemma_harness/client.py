# gemma_harness/client.py  (checked against google-genai 2.33.0)
import time, random, json, hashlib, difflib, re, inspect, typing
import concurrent.futures as cf
from typing import Callable, Any
from google import genai
from google.genai import types, errors

FAIL_REASONS = {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL",
                "TOO_MANY_TOOL_CALLS", "OTHER", "FINISH_REASON_UNSPECIFIED"}

class Client:
    def __init__(self, model="gemma-4-26b-a4b-it", api_key=None, max_steps=8,
                 max_retries=1, tool_timeout_s=10.0, route_thinking=True,
                 system_instruction="", on_event: Callable[[dict], None] | None = None):
        self.raw = genai.Client(api_key=api_key)   # reads GEMINI_API_KEY if None
        self.model, self.max_steps, self.max_retries = model, max_steps, max_retries
        self.tool_timeout_s, self.route_thinking = tool_timeout_s, route_thinking
        self.system_instruction = system_instruction
        self.trace: list[dict] = []
        self.on_event = on_event

    def emit(self, step: int, label: str, **data):
        ev = {"t": round(time.time(), 3), "step": step, "label": label, **data}
        self.trace.append(ev)
        if self.on_event: self.on_event(ev)

    def _generate(self, contents, decls, thinking: str, step: int, extra_sys: str = ""):
        cfg = types.GenerateContentConfig(
            system_instruction=(self.system_instruction + extra_sys) or None,
            tools=[types.Tool(function_declarations=decls)] if decls else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level=thinking),
        )
        for attempt in range(5):
            try:
                t0 = time.time()
                r = self.raw.models.generate_content(model=self.model, contents=contents, config=cfg)
                um = r.usage_metadata
                self.emit(step, "CALL", ms=int((time.time() - t0) * 1000), thinking=thinking,
                          in_tok=getattr(um, "prompt_token_count", None),
                          out_tok=getattr(um, "candidates_token_count", None),
                          think_tok=getattr(um, "thoughts_token_count", None))
                return r
            except errors.APIError as e:
                if e.code in (429, 500, 503) and attempt < 4:
                    wait = min(30, 2 ** attempt) + random.random()
                    self.emit(step, "BACKOFF_429" if e.code == 429 else "BACKOFF_5XX", wait_s=round(wait, 1))
                    time.sleep(wait); continue
                raise

    def run(self, prompt: str, tools: list[Callable] = ()) -> dict:
        self.trace = []
        tool_map = {f.__name__: f for f in tools}
        decls = [types.FunctionDeclaration.from_callable(client=self.raw, callable=f) for f in tools]
        extra = lint_prompt(self.system_instruction + "\n" + prompt, tool_map, self)
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        seen: dict[str, Any] = {}
        for step in range(1, self.max_steps + 1):
            thinking = "minimal" if (self.route_thinking and step < self.max_steps) else "high"
            r = self._generate(contents, decls, thinking, step, extra)
            r, _ = recover_blank(self, r, contents, decls, step, extra)
            calls = r.function_calls or []
            if not calls:
                text = (r.text or "").strip()
                if not text:
                    self.emit(step, "SILENT_FAIL_PREVENTED")
                    text = fallback_text(tool_map)
                self.emit(step, "FINAL", chars=len(text))
                return {"text": text, "trace": self.trace, "steps": step}
            contents.append(r.candidates[0].content)
            parts = []
            for fc in calls:
                name, args = fix_tool_name(self, fc.name, tool_map, step), dict(fc.args or {})
                if name is None:
                    parts.append(types.Part.from_function_response(name=fc.name,
                        response={"error": f"unknown tool. available: {list(tool_map)}"}))
                    continue
                args = fix_args(self, tool_map[name], args, step)
                key = hashlib.md5(f"{name}{json.dumps(args, sort_keys=True)}".encode()).hexdigest()
                if key in seen:
                    self.emit(step, "DEDUPED", tool=name)
                    parts.append(types.Part.from_function_response(name=name, response={
                        "result": seen[key],
                        "note": "ALREADY CALLED with same args. Use this result; call a different tool or answer."}))
                    continue
                result = call_tool(self, tool_map[name], args, step)
                seen[key] = result
                parts.append(types.Part.from_function_response(name=name, response={"result": result}))
            contents.append(types.Content(role="user", parts=parts))
        self.emit(self.max_steps, "STEP_CAP")
        return {"text": fallback_text(tool_map), "trace": self.trace, "steps": self.max_steps}

# ---------- recovery functions ----------
def lint_prompt(text: str, tool_map: dict, h: Client) -> str:
    mentioned = set(re.findall(r"\b([a-z]+_[a-z_]+)\b", text))
    phantom = sorted(m for m in mentioned if m not in tool_map and m.split("_")[0] in
                     {"get", "search", "send", "fetch", "lookup", "create", "delete", "web"})
    if phantom:
        h.emit(0, "PHANTOM_TOOL_LINTED", names=phantom)
        return (f"\nTools NOT available this turn: {phantom}. Only call: {list(tool_map)}."
                " If none fits, answer from your own knowledge and say so.")
    return ""

def recover_blank(h: Client, r, contents, decls, step, extra):
    cand = r.candidates[0] if r.candidates else None
    fr = str(getattr(cand, "finish_reason", "") or "").split(".")[-1]
    blank = not r.function_calls and not (r.text or "").strip()
    if fr in FAIL_REASONS or blank:
        for i in range(h.max_retries):
            h.emit(step, "RETRIED", reason=fr or "EMPTY", attempt=i + 1)
            hint = (f"\nYour previous reply failed ({fr or 'empty'}). Call exactly one of "
                    f"{[d.name for d in decls]} with valid JSON args, or answer in plain text.")
            r = h._generate(contents, decls, "minimal", step, extra + hint)
            if r.function_calls or (r.text or "").strip():
                return r, True
    return r, False

def fix_tool_name(h: Client, name: str, tool_map: dict, step: int):
    if name in tool_map: return name
    m = difflib.get_close_matches(name, list(tool_map), n=1, cutoff=0.6)
    h.emit(step, "TOOL_NAME_FIXED" if m else "UNKNOWN_TOOL", got=name, used=m[0] if m else None)
    return m[0] if m else None

def fix_args(h: Client, fn: Callable, args: dict, step: int) -> dict:
    sig, hints = inspect.signature(fn), typing.get_type_hints(fn)
    out, fixed = {}, []
    for k, v in args.items():
        if k not in sig.parameters:
            m = difflib.get_close_matches(k, list(sig.parameters), n=1, cutoff=0.6)
            if not m: fixed.append(f"drop:{k}"); continue
            fixed.append(f"{k}->{m[0]}"); k = m[0]
        t = hints.get(k)
        try:
            if t in (int, float) and isinstance(v, str): v = t(v.strip()); fixed.append(f"coerce:{k}")
            elif t is bool and isinstance(v, str): v = v.strip().lower() in ("true", "1", "yes"); fixed.append(f"coerce:{k}")
        except ValueError: pass
        out[k] = v
    missing = [p for p, s in sig.parameters.items() if s.default is s.empty and p not in out]
    if fixed: h.emit(step, "ARGS_COERCED", fixes=fixed)
    if missing: h.emit(step, "ARGS_MISSING", missing=missing)
    return out

def call_tool(h: Client, fn: Callable, args: dict, step: int):
    ex = cf.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn, **args)
    try:
        res = fut.result(timeout=h.tool_timeout_s)
        h.emit(step, "TOOL_OK", tool=fn.__name__)
        return res
    except cf.TimeoutError:
        h.emit(step, "TOOL_TIMEOUT", tool=fn.__name__)
        return {"error": f"{fn.__name__} timed out after {h.tool_timeout_s}s"}
    except Exception as e:
        h.emit(step, "TOOL_ERROR", tool=fn.__name__, err=repr(e)[:200])
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        ex.shutdown(wait=False)   # don't block on a hung tool thread

def fallback_text(tool_map: dict) -> str:
    return ("Sorry, I couldn't finish this with my tools "
            f"(available: {', '.join(tool_map) or 'none'}). Here is what I know without them.")
