# gemma-harness 🛠️

A drop-in wrapper for the `google-genai` SDK that ensures Gemma 4 agents never fail silently.

## 🚀 The Problem
When building tool-using agents with Gemma 4, you often encounter "silent failures":
- **Blank Replies:** The model returns an empty response due to a `MALFORMED_FUNCTION_CALL`.
- **Phantom Tools:** The model tries to call a tool mentioned in the prompt but not provided in the code.
- **Fragile Args:** A slight misspelling in a tool argument causes the entire agent loop to crash.
- **Infinite Loops:** The agent calls the same tool with the same arguments repeatedly.

## ✅ The Solution
`gemma-harness` intercepts the agent loop and applies a recovery taxonomy:

| Failure | Recovery | Trace Label |
|---|---|---|
| Phantom Tool | Lints prompt and warns model before the call | `PHANTOM_TOOL_LINTED` |
| Blank/Malformed | Retries once with a format hint | `RETRIED` |
| Wrong Tool Name | Fuzzy-matches the tool name (cutoff 0.6) | `TOOL_NAME_FIXED` |
| Tool Crash/Timeout | Catches exception and feeds error back to model | `TOOL_ERROR` |
| Repeat Loop | Caches result and forces a new step | `DEDUPED` |
| API 429/5xx | Exponential backoff with jitter | `BACKOFF` |

## 📦 Quick Start

```python
from gemma_harness.client import Client

# One-line swap: replace genai.Client with gemma_harness.Client
h = Client(api_key="YOUR_KEY", model="gemma-4-26b-a4b-it")

# Define your tools as standard Python functions
def get_weather(city: str):
    return {"temp": 31}

# Run the agent
result = h.run("What is the weather in Karachi?", tools=[get_weather])
print(result["text"])