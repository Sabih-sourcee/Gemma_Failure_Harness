import time, random, json, hashlib, difflib, concurrent.futures as cf
from typing import Callable, Any
from google import genai
from google.genai import types, errors

FAIL_REASONS = {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS", "OTHER"}

class Client:
    def __init__(self, model="gemma-4-26b-a4b-it", api_key=None, max_steps=8,
                 max_retries=1, tool_timeout_s=10.0, route_thinking=True,
                 system_instruction="", on_event=None):
        self.raw = genai.Client(api_key=api_key)
        self.model, self.max_steps, self.max_retries = model, max_steps, max_retries
        self.tool_timeout_s, self.route_thinking = tool_timeout_s, route_thinking
        self.system_instruction = system_instruction
        self.trace = []
        self.on_event = on_event

    def emit(self, step: int, label: str, **data):
        ev = {"t": round(time.time(), 3), "step": step, "label": label, **data}
        self.trace.append(ev)
        if self.on_event: self.on_event(ev)

    def _generate(self, contents, decls, thinking, step, extra_sys=""):
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
                self.emit(step, "CALL", ms=int((time.time()-t0)*1000), thinking=thinking)
                return r
            except errors.APIError as e:
                if e.code in (429, 500, 503) and attempt < 4:
                    wait = min(30, 2 ** attempt) + random.random()
                    self.emit(step, "BACKOFF", wait_s=round(wait, 1))
                    time.sleep(wait); continue
                raise

    def run(self, prompt: str, tools: list[Callable] = ()) -> dict:
        self.trace = []
        tool_map = {f.__name__: f for f in tools}
        decls = [types.FunctionDeclaration.from_callable(client=self.raw, callable=f) for f in tools]
        
        # Phantom tool linting
        extra = ""
        import re
        mentioned = set(re.findall(r"\b([a-z]+_[a-z_]+)\b", prompt))
        phantom = [m for m in mentioned if m not in tool_map and any(x in m for x in ["get", "search", "send", "web"])]
        if phantom:
            self.emit(0, "PHANTOM_TOOL_LINTED", names=phantom)
            extra = f"\nTools NOT available: {phantom}. Only use: {list(tool_map)}."

        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        seen = {}

        for step in range(1, self.max_steps + 1):
            thinking = "minimal" if (self.route_thinking and step < self.max_steps) else "high"
            r = self._generate(contents, decls, thinking, step, extra)
            
            # Recover blank/malformed
            cand = r.candidates[0] if r.candidates else None
            fr = str(getattr(cand, "finish_reason", "")).split(".")[-1]
            if (not r.function_calls and not (r.text or "").strip()) or fr in FAIL_REASONS:
                self.emit(step, "RETRIED", reason=fr)
                r = self._generate(contents, decls, "minimal", step, extra + "\nFix your tool call format.")

            if not r.function_calls:
                text = (r.text or "").strip()
                if not text:
                    self.emit(step, "SILENT_FAIL_PREVENTED")
                    text = "I encountered an error but I'm here to help."
                self.emit(step, "FINAL", chars=len(text))
                return {"text": text, "trace": self.trace}

            contents.append(r.candidates[0].content)
            parts = []
            for fc in r.function_calls:
                # 1. Fix Tool Name
                name = fc.name if fc.name in tool_map else difflib.get_close_matches(fc.name, list(tool_map), n=1, cutoff=0.6)
                name = name[0] if isinstance(name, list) and name else (fc.name if fc.name in tool_map else None)
                if name != fc.name: self.emit(step, "TOOL_NAME_FIXED", got=fc.name, used=name)

                if not name:
                    parts.append(types.Part.from_function_response(name=fc.name, response={"error": "Unknown tool"}))
                    continue

                # 2. Fix Args (Simple coercion)
                args = dict(fc.args or {})
                # 3. Dedupe
                key = hashlib.md5(f"{name}{json.dumps(args, sort_keys=True)}".encode()).hexdigest()
                if key in seen:
                    self.emit(step, "DEDUPED", tool=name)
                    parts.append(types.Part.from_function_response(name=name, response={"result": seen[key], "note": "already called"}))
                    continue

                # 4. Call Tool with Timeout
                try:
                    with cf.ThreadPoolExecutor().submit(tool_map[name], **args).result(timeout=self.tool_timeout_s) as res:
                        self.emit(step, "TOOL_OK", tool=name)
                        seen[key] = res
                        parts.append(types.Part.from_function_response(name=name, response={"result": res}))
                except Exception as e:
                    self.emit(step, "TOOL_ERROR", tool=name, err=str(e))
                    parts.append(types.Part.from_function_response(name=name, response={"error": str(e)}))

            contents.append(types.Content(role="user", parts=parts))
        
        return {"text": "Step limit reached", "trace": self.trace}