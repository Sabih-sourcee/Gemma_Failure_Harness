# gemma-harness

**A safety net for Gemma 4 agents.** When Gemma tries to use a tool and something goes wrong, this fixes it or tells you clearly what broke, instead of failing silently.

**Live lab:** [gemma-harness.vercel.app](https://gemma-harness.vercel.app)  
**Repo:** [Sabih-sourcee/Gemma_Failure_Harness](https://github.com/Sabih-sourcee/Gemma_Failure_Harness)

---

## What problem does this solve?

An "agent" is an AI that can use **tools**, which are small functions like "get the weather" or "add two numbers". The AI decides which tool to call and what to pass it.

Sometimes that goes wrong:

- It calls a tool that **doesn't exist**
- It sends a **broken or empty** request
- It passes `"5"` (text) when the tool needs `5` (a number)
- A tool is **slow** and everything freezes
- Google says **"too many requests"** (error 429)
- It calls the **same tool again and again**

Usually you just get a crash or an empty answer, with no idea why.

**gemma-harness** sits between Gemma and your tools. It catches each of these problems, fixes what it can, and writes down every fix it made.

```
You  →  gemma-harness  →  Gemma 4
              ↓
           Your tools
```

---

## Easiest way: run it in Google Colab (no install)

You only need a Google account.

### Step 1: Get a free API key
1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click **Create API key**
3. Copy it. **Never share it or paste it into code.**

### Step 2: Open the notebook
1. Open `gemma_harness_demo.ipynb` in [Google Colab](https://colab.research.google.com/) (File → Upload notebook, or use the Colab link in this repo)

### Step 3: Add your key safely
1. In Colab, click the **🔑 key icon** on the left sidebar (Secrets)
2. Click **Add new secret**
3. Name: `GEMINI_API_KEY`
4. Value: paste your key
5. Turn on **Notebook access**

### Step 4: Run it
1. Click **Runtime → Run all**
2. Wait a few minutes

At the end you'll see a table comparing:
- **Plain agent:** no safety net
- **With gemma-harness:** with the safety net

Want to try your own question? Scroll to the last cell, **"Try your own"**, type your prompt, and run it.

---

## Run it on your own computer

You need **Python 3.10 or newer**. Check with:
```bash
python --version
```

**1. Download the code**
```bash
git clone https://github.com/Sabih-sourcee/Gemma_Failure_Harness.git
cd Gemma_Failure_Harness
```

**2. Install what it needs**
```bash
pip install -r requirements.txt
```

**3. Set your API key** (only for this terminal window)

Mac / Linux:
```bash
export GEMINI_API_KEY="your-key-here"
```
Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="your-key-here"
```

Or paste into `.env` at the project root:
```
GEMINI_API_KEY=your-key-here
```

**4. Run the comparison**
```bash
python3 bench.py
```

**5. Run the live UI**
```bash
cd web && npm install && npm run build && cd ..
python3 ui_server.py
# open http://127.0.0.1:3847
```

---

## Use it in your own project

```python
from gemma_harness import Client
from demo.tools import get_weather, add

client = Client(
    model="gemma-4-26b-a4b-it",
    on_event=lambda ev: print(ev["label"], ev),
)
result = client.run("What's the weather in Karachi?", tools=[get_weather, add])

print(result["text"])   # the final answer
print(result["trace"])  # every fix the harness made
```

That's it. Write normal Python functions, give them to `Client`, and ask your question.

---

## What it fixes

| Problem | What gemma-harness does |
|---|---|
| Calls a tool that doesn't exist | Fuzzy-fixes the name or tells Gemma the real tool list |
| Broken or empty tool call | Retries once, then falls back to a plain text answer |
| Wrong type (`"5"` instead of `5`) | Converts it before running the tool |
| Tool is too slow | Stops it after a timeout and tells Gemma |
| Tool crashes | Catches the error and passes it back as a readable message |
| Same call over and over | Stops the loop |
| Too many requests (429) | Waits a bit and tries again |

Every fix is saved in `result["trace"]`, so you can always see what happened.

---

## Results (live measured)

We ran stress prompts on `gemma-4-26b-a4b-it` (37 plain + 37 harness live API runs):

| | Plain loop | gemma-harness | Reduction |
|---|---|---|---|
| Silent failures | **5/37 (13.5%)** | **1/37 (2.7%)** | **80%** |
| Unsuccessful | **14/37 (37.8%)** | **2/37 (5.4%)** | **86%** |
| Avg latency | **10.5s** | **3.2s** | — |
| Crashes | **9** | **0** | — |

Raw data: `results.csv` · Summary: `BENCH_SUMMARY.txt`

These prompts are built to make tool calls fail on purpose, so this is a hard test, not everyday use.

---

## Common problems

**"API key not found"**
Check that the secret is named exactly `GEMINI_API_KEY` and that Notebook access is turned on (Colab), or that `.env` / Vercel env has the key.

**"429 / Resource exhausted"**
You're on the free tier and sent too many requests. Wait a minute and run it again. The harness already slows down on its own.

**"ModuleNotFoundError"**
Run `pip install -r requirements.txt` again.

---

## Built with
Python · google-genai SDK · Gemma 4 via the Gemini API · Google AI Studio · React · Vercel

Built at MLH Hacktoberfest Hack Day Karachi, Oct 2026.

## License
MIT. Free to use, change, and share.
