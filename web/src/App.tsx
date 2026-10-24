import { useEffect, useRef, useState } from "react";

type Side = "plain" | "harness";

type TraceEvent = {
  label: string;
  detail?: string;
  delayMs: number;
};

type Scenario = {
  id: string;
  title: string;
  problem: string;
  prompt: string;
  modelDid: string;
  plain: {
    events: TraceEvent[];
    outcome: "crash" | "empty" | "wrong" | "ok";
    answer: string;
  };
  harness: {
    events: TraceEvent[];
    outcome: "recovered" | "ok";
    answer: string;
  };
};

const SCENARIOS: Scenario[] = [
  {
    id: "typo",
    title: "Typo tool name",
    problem: "Model invents get_wether instead of get_weather",
    prompt: "What's the weather in Lahore? Use get_weather_now.",
    modelDid: 'function_call name="get_wether" args={citty: "Lahore"}',
    plain: {
      events: [
        { label: "CALL", detail: "model returns tool call", delayMs: 400 },
        { label: "CRASH", detail: "KeyError: unknown tool get_wether", delayMs: 700 },
      ],
      outcome: "crash",
      answer: "(process died — user sees nothing useful)",
    },
    harness: {
      events: [
        { label: "PHANTOM_TOOL_LINTED", detail: "get_weather_now not in tool list", delayMs: 350 },
        { label: "CALL", detail: "model still typos get_wether", delayMs: 650 },
        { label: "TOOL_NAME_FIXED", detail: "get_wether → get_weather", delayMs: 450 },
        { label: "ARGS_COERCED", detail: "citty → city", delayMs: 400 },
        { label: "TOOL_OK", detail: "get_weather(Lahore)", delayMs: 500 },
        { label: "FINAL", detail: "answer returned", delayMs: 450 },
      ],
      outcome: "recovered",
      answer: "Lahore is 31°C with no rain.",
    },
  },
  {
    id: "strings",
    title: "Wrong arg types",
    problem: 'Model passes "7" and "8" as strings to a float tool',
    prompt: 'Add "7" and "8" with the add tool.',
    modelDid: 'function_call name="add" args={a: "7", b: "8"}',
    plain: {
      events: [
        { label: "CALL", detail: "model returns add(\"7\",\"8\")", delayMs: 400 },
        { label: "CRASH", detail: "TypeError / bad coerce in raw call", delayMs: 700 },
      ],
      outcome: "crash",
      answer: "(crash or silent empty — no number for the user)",
    },
    harness: {
      events: [
        { label: "CALL", detail: "model returns string args", delayMs: 400 },
        { label: "ARGS_COERCED", detail: '"7"→7.0, "8"→8.0', delayMs: 500 },
        { label: "TOOL_OK", detail: "add → 15", delayMs: 450 },
        { label: "FINAL", detail: "answer returned", delayMs: 400 },
      ],
      outcome: "recovered",
      answer: "7 + 8 = 15.",
    },
  },
  {
    id: "crash",
    title: "Tool exception",
    problem: "flaky_db raises ConnectionError mid-turn",
    prompt: "Fetch user 42 with flaky_db and tell me their name.",
    modelDid: "function_call name=\"flaky_db\" args={user_id: 42}",
    plain: {
      events: [
        { label: "CALL", detail: "model calls flaky_db", delayMs: 400 },
        { label: "CRASH", detail: "ConnectionError: db down", delayMs: 650 },
      ],
      outcome: "crash",
      answer: "(uncaught exception — whole agent stops)",
    },
    harness: {
      events: [
        { label: "CALL", detail: "model calls flaky_db", delayMs: 400 },
        { label: "TOOL_ERROR", detail: "caught ConnectionError('db down')", delayMs: 550 },
        { label: "CALL", detail: "error fed back to model", delayMs: 700 },
        { label: "FINAL", detail: "honest answer, no crash", delayMs: 450 },
      ],
      outcome: "recovered",
      answer:
        "I couldn't fetch user 42 — the database is down (ConnectionError).",
    },
  },
  {
    id: "phantom",
    title: "Phantom tool",
    problem: "Prompt names web_search which was never declared",
    prompt: "Use web_search to find today's news in Karachi.",
    modelDid: "tries to call web_search (not in tools)",
    plain: {
      events: [
        { label: "CALL", detail: "model calls web_search", delayMs: 450 },
        { label: "CRASH", detail: "unknown tool — KeyError", delayMs: 650 },
      ],
      outcome: "crash",
      answer: "(blank or crash)",
    },
    harness: {
      events: [
        { label: "PHANTOM_TOOL_LINTED", detail: "web_search not available", delayMs: 400 },
        { label: "CALL", detail: "model steered away from phantom", delayMs: 700 },
        { label: "FINAL", detail: "answers from knowledge + says so", delayMs: 500 },
      ],
      outcome: "recovered",
      answer:
        "web_search isn't available. Here's what I can say without live search…",
    },
  },
  {
    id: "loop",
    title: "Repeat loop",
    problem: "Model keeps calling the same weather check forever",
    prompt: "Keep calling get_weather for Karachi until it says it's raining.",
    modelDid: "get_weather(Karachi) × N identical calls",
    plain: {
      events: [
        { label: "CALL", detail: "get_weather Karachi", delayMs: 350 },
        { label: "TOOL_OK", detail: "rain=false", delayMs: 400 },
        { label: "CALL", detail: "same call again…", delayMs: 400 },
        { label: "TOOL_OK", detail: "rain=false", delayMs: 350 },
        { label: "CALL", detail: "loop continues / step cap", delayMs: 400 },
        { label: "EMPTY", detail: "hits step limit with no answer", delayMs: 500 },
      ],
      outcome: "empty",
      answer: "(empty — silent failure)",
    },
    harness: {
      events: [
        { label: "CALL", detail: "get_weather Karachi", delayMs: 350 },
        { label: "TOOL_OK", detail: "rain=false", delayMs: 400 },
        { label: "DEDUPED", detail: "same args — reuse cached result", delayMs: 450 },
        { label: "DEDUPED", detail: "loop broken", delayMs: 400 },
        { label: "FINAL", detail: "tells user it is not raining", delayMs: 450 },
      ],
      outcome: "recovered",
      answer: "Karachi is 31°C and not raining. Repeated checks returned the same result.",
    },
  },
];

const LABEL_TONE: Record<string, string> = {
  TOOL_NAME_FIXED: "ember",
  ARGS_COERCED: "ember",
  TOOL_ERROR: "warn",
  TOOL_TIMEOUT: "warn",
  PHANTOM_TOOL_LINTED: "sea",
  RETRIED: "sea",
  DEDUPED: "sea",
  SILENT_FAIL_PREVENTED: "warn",
  CRASH: "warn",
  EMPTY: "warn",
  FINAL: "ok",
  TOOL_OK: "ok",
  CALL: "muted",
};

function toneClass(label: string): string {
  const t = LABEL_TONE[label] || "muted";
  if (t === "ember")
    return "border-[var(--color-ember)]/40 bg-[var(--color-ember)]/15 text-[var(--color-ember)]";
  if (t === "warn") return "border-red-500/30 bg-red-500/10 text-red-700";
  if (t === "sea")
    return "border-[var(--color-sea)]/35 bg-[var(--color-sea)]/10 text-[var(--color-moss)]";
  if (t === "ok") return "border-emerald-600/30 bg-emerald-600/10 text-emerald-800";
  return "border-black/10 bg-black/5 text-[var(--color-ink)]/65";
}

function OutcomeBadge({
  outcome,
}: {
  outcome: Scenario["plain"]["outcome"] | Scenario["harness"]["outcome"];
}) {
  const map: Record<string, { text: string; cls: string }> = {
    crash: { text: "CRASH", cls: "bg-red-600 text-white" },
    empty: { text: "SILENT FAIL", cls: "bg-red-700 text-white" },
    wrong: { text: "WRONG", cls: "bg-amber-700 text-white" },
    ok: { text: "OK", cls: "bg-emerald-700 text-white" },
    recovered: { text: "RECOVERED", cls: "bg-[var(--color-sea)] text-white" },
  };
  const m = map[outcome];
  return (
    <span className={`rounded px-2 py-0.5 font-[family-name:var(--font-mono)] text-[10px] tracking-wide ${m.cls}`}>
      {m.text}
    </span>
  );
}

export default function App() {
  const [activeId, setActiveId] = useState(SCENARIOS[0].id);
  const [playing, setPlaying] = useState(false);
  const [plainShown, setPlainShown] = useState<TraceEvent[]>([]);
  const [harnessShown, setHarnessShown] = useState<TraceEvent[]>([]);
  const [plainDone, setPlainDone] = useState(false);
  const [harnessDone, setHarnessDone] = useState(false);
  const [phase, setPhase] = useState<"idle" | "plain" | "harness" | "done">("idle");
  const timers = useRef<number[]>([]);

  const scenario = SCENARIOS.find((s) => s.id === activeId) || SCENARIOS[0];

  function clearTimers() {
    timers.current.forEach((t) => window.clearTimeout(t));
    timers.current = [];
  }

  function playScenario(next?: Scenario) {
    const s = next || scenario;
    clearTimers();
    setPlaying(true);
    setPlainShown([]);
    setHarnessShown([]);
    setPlainDone(false);
    setHarnessDone(false);
    setPhase("plain");

    let t = 300;
    s.plain.events.forEach((ev, i) => {
      t += ev.delayMs;
      const id = window.setTimeout(() => {
        setPlainShown((prev) => [...prev, ev]);
        if (i === s.plain.events.length - 1) {
          setPlainDone(true);
          setPhase("harness");
        }
      }, t);
      timers.current.push(id);
    });

    t += 600;
    s.harness.events.forEach((ev, i) => {
      t += ev.delayMs;
      const id = window.setTimeout(() => {
        setHarnessShown((prev) => [...prev, ev]);
        if (i === s.harness.events.length - 1) {
          setHarnessDone(true);
          setPhase("done");
          setPlaying(false);
        }
      }, t);
      timers.current.push(id);
    });
  }

  useEffect(() => {
    playScenario(scenario);
    return clearTimers;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId]);

  return (
    <div className="mx-auto min-h-screen max-w-6xl px-4 pb-16 pt-8 sm:px-6 sm:pt-10">
      <header className="rise relative overflow-hidden pb-10">
        <div
          className="pointer-events-none absolute inset-0 -z-10 opacity-90"
          style={{
            background:
              "radial-gradient(ellipse 80% 60% at 70% 20%, rgba(31,111,106,0.22), transparent 60%), radial-gradient(ellipse 50% 40% at 10% 80%, rgba(224,122,61,0.14), transparent 55%)",
          }}
        />
        <p className="font-[family-name:var(--font-mono)] text-xs tracking-[0.22em] text-[var(--color-sea)] uppercase">
          Failure harness · Gemma 4
        </p>
        <h1 className="mt-3 font-[family-name:var(--font-display)] text-5xl leading-[0.95] tracking-tight text-[var(--color-moss)] sm:text-6xl">
          gemma-harness
        </h1>
        <p className="mt-4 max-w-xl text-lg text-[var(--color-ink)]/70">
          Watch how a normal agent loop hides model mistakes — and how the harness
          labels, fixes, or surfaces them instead of failing silently.
        </p>
      </header>

      <section className="rise mb-8" style={{ animationDelay: "60ms" }}>
        <p className="mb-3 text-sm font-medium text-[var(--color-moss)]">
          Pick a model failure
        </p>
        <div className="flex flex-wrap gap-2">
          {SCENARIOS.map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() => setActiveId(s.id)}
              className={`rounded-lg px-3 py-1.5 text-sm transition ${
                activeId === s.id
                  ? "bg-[var(--color-moss)] text-[var(--color-sand)]"
                  : "bg-white/60 text-[var(--color-moss)] hover:bg-white"
              }`}
            >
              {s.title}
            </button>
          ))}
        </div>
      </section>

      <section
        className="rise mb-6 grid gap-4 lg:grid-cols-[1fr_1.15fr]"
        style={{ animationDelay: "100ms" }}
      >
        <div className="space-y-4">
          <div>
            <h2 className="font-[family-name:var(--font-display)] text-2xl text-[var(--color-moss)]">
              What the model did wrong
            </h2>
            <p className="mt-2 text-[var(--color-ink)]/70">{scenario.problem}</p>
          </div>

          <div>
            <p className="mb-1 font-[family-name:var(--font-mono)] text-[10px] tracking-wider text-[var(--color-ink)]/45 uppercase">
              User prompt
            </p>
            <p className="font-[family-name:var(--font-mono)] text-sm leading-relaxed text-[var(--color-ink)]/85">
              {scenario.prompt}
            </p>
          </div>

          <div>
            <p className="mb-1 font-[family-name:var(--font-mono)] text-[10px] tracking-wider text-[var(--color-ink)]/45 uppercase">
              Model behaviour
            </p>
            <p className="rounded-lg border border-[var(--color-ember)]/25 bg-[var(--color-ember)]/8 px-3 py-2 font-[family-name:var(--font-mono)] text-xs text-[var(--color-ember)]">
              {scenario.modelDid}
            </p>
          </div>

          <button
            type="button"
            disabled={playing}
            onClick={() => playScenario()}
            className="rounded-xl bg-[var(--color-ember)] px-5 py-2.5 text-sm font-semibold text-white transition hover:brightness-105 disabled:opacity-50"
          >
            {playing ? "Replaying…" : "Replay detection"}
          </button>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <TracePanel
            side="plain"
            title="Plain agent loop"
            subtitle="No recovery"
            events={plainShown}
            done={plainDone}
            outcome={scenario.plain.outcome}
            answer={scenario.plain.answer}
            active={phase === "plain" || (phase === "harness" && plainDone)}
          />
          <TracePanel
            side="harness"
            title="gemma-harness"
            subtitle="Detect → label → fix"
            events={harnessShown}
            done={harnessDone}
            outcome={scenario.harness.outcome}
            answer={scenario.harness.answer}
            active={phase === "harness" || phase === "done"}
            accent
          />
        </div>
      </section>

      <section className="rise mt-12 border-t border-[var(--color-line)]/80 pt-10" style={{ animationDelay: "140ms" }}>
        <h2 className="font-[family-name:var(--font-display)] text-2xl text-[var(--color-moss)]">
          Live measured on Gemma 4
        </h2>
        <p className="mt-2 max-w-2xl text-sm text-[var(--color-ink)]/65">
          Real API runs against <code className="font-[family-name:var(--font-mono)] text-xs">gemma-4-26b-a4b-it</code>
          {" "}— 37 plain vs 37 harness stress turns. Not simulated.
        </p>
        <div className="mt-6 grid gap-6 sm:grid-cols-3">
          <Stat
            label="Silent failures"
            plain="13.5%"
            harness="2.7%"
            note="80% fewer blank/hidden fails"
          />
          <Stat
            label="Unsuccessful turns"
            plain="37.8%"
            harness="5.4%"
            note="86% reduction"
          />
          <Stat
            label="Avg latency"
            plain="10.5s"
            harness="3.2s"
            note="9 crashes → 0 with harness"
          />
        </div>
      </section>

      <section className="rise mt-12" style={{ animationDelay: "180ms" }}>
        <h2 className="font-[family-name:var(--font-display)] text-2xl text-[var(--color-moss)]">
          Detection labels
        </h2>
        <p className="mt-2 mb-5 max-w-2xl text-sm text-[var(--color-ink)]/65">
          Every recovery is named in the trace so you can see exactly which model
          behaviour triggered it.
        </p>
        <ul className="grid gap-3 sm:grid-cols-2">
          {[
            ["TOOL_NAME_FIXED", "Fuzzy-match a mistyped tool name"],
            ["ARGS_COERCED", "Rename keys / coerce \"10\" → 10"],
            ["TOOL_ERROR", "Catch tool exceptions; tell the model"],
            ["TOOL_TIMEOUT", "Kill hung tools; return a clear error"],
            ["PHANTOM_TOOL_LINTED", "Prompt named a tool you never sent"],
            ["DEDUPED", "Stop identical call loops"],
            ["RETRIED", "Blank / malformed finish_reason → one retry"],
            ["BACKOFF_429", "Rate-limit backoff instead of dying"],
          ].map(([lab, desc]) => (
            <li key={lab} className="flex items-start gap-3">
              <span
                className={`mt-0.5 shrink-0 rounded border px-1.5 py-0.5 font-[family-name:var(--font-mono)] text-[10px] ${toneClass(lab)}`}
              >
                {lab}
              </span>
              <span className="text-sm text-[var(--color-ink)]/75">{desc}</span>
            </li>
          ))}
        </ul>
      </section>

      <footer className="mt-14 border-t border-[var(--color-line)]/70 pt-6 text-sm text-[var(--color-ink)]/50">
        MLH Hacktoberfest Hack Day Karachi ·{" "}
        <a
          className="text-[var(--color-sea)] underline-offset-2 hover:underline"
          href="https://github.com/Sabih-sourcee/Gemma_Failure_Harness"
        >
          GitHub
        </a>
      </footer>
    </div>
  );
}

function TracePanel({
  title,
  subtitle,
  events,
  done,
  outcome,
  answer,
  accent,
  active,
}: {
  side: Side;
  title: string;
  subtitle: string;
  events: TraceEvent[];
  done: boolean;
  outcome: Scenario["plain"]["outcome"] | Scenario["harness"]["outcome"];
  answer: string;
  accent?: boolean;
  active?: boolean;
}) {
  const endRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [events.length]);

  return (
    <div
      className={`flex min-h-[340px] flex-col rounded-2xl border p-4 transition ${
        accent
          ? "border-[var(--color-sea)]/35 bg-[var(--color-moss)] text-[var(--color-sand)]"
          : "border-[var(--color-line)] bg-white/70"
      } ${active ? "ring-2 ring-[var(--color-ember)]/35" : ""}`}
    >
      <div className="mb-3 flex items-start justify-between gap-2">
        <div>
          <h3
            className={`font-[family-name:var(--font-display)] text-lg ${
              accent ? "text-[var(--color-sand)]" : "text-[var(--color-moss)]"
            }`}
          >
            {title}
          </h3>
          <p className={`text-xs ${accent ? "text-[var(--color-sand)]/55" : "text-[var(--color-ink)]/45"}`}>
            {subtitle}
          </p>
        </div>
        {done && <OutcomeBadge outcome={outcome} />}
      </div>

      <div
        className={`mb-3 max-h-[200px] flex-1 space-y-1.5 overflow-y-auto font-[family-name:var(--font-mono)] text-[11px] ${
          accent ? "text-[var(--color-sand)]/80" : "text-[var(--color-ink)]/70"
        }`}
      >
        {events.length === 0 && (
          <p className={accent ? "text-[var(--color-sand)]/40" : "text-[var(--color-ink)]/35"}>
            Waiting for model turn…
          </p>
        )}
        {events.map((ev, i) => (
          <div key={`${ev.label}-${i}`} className="stream-in flex flex-wrap items-baseline gap-2">
            <span className={`rounded border px-1.5 py-0.5 ${toneClass(ev.label)}`}>
              {ev.label}
            </span>
            {ev.detail && <span className="opacity-70">{ev.detail}</span>}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      {done && (
        <div
          className={`stream-in border-t pt-3 text-sm leading-relaxed ${
            accent
              ? "border-white/10 text-[var(--color-sand)]/90"
              : "border-[var(--color-line)] text-[var(--color-ink)]/80"
          }`}
        >
          {answer}
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  plain,
  harness,
  note,
}: {
  label: string;
  plain: string;
  harness: string;
  note: string;
}) {
  return (
    <div>
      <p className="font-[family-name:var(--font-mono)] text-[10px] tracking-wider text-[var(--color-ink)]/45 uppercase">
        {label}
      </p>
      <p className="mt-2 font-[family-name:var(--font-display)] text-3xl text-[var(--color-moss)]">
        <span className="text-red-700/80">{plain}</span>
        <span className="mx-2 text-lg text-[var(--color-ink)]/30">→</span>
        <span className="text-[var(--color-sea)]">{harness}</span>
      </p>
      <p className="mt-1 text-xs text-[var(--color-ink)]/55">{note}</p>
    </div>
  );
}
