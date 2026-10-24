import os
from dotenv import load_dotenv # 1. Import the helpe
from tools import get_weather, add, save_note, slow_lookup, flaky_db
from gemma_harness.client import Client
from raw_agent import raw_run  # Import the baseline we built in Step 1

load_dotenv() # 2. This reads the .env file and loads the keys into memory

MY_KEY = os.environ.get("GEMINI_API_KEY")
my_tools = [get_weather, add, save_note, slow_lookup, flaky_db]

PROMPTS = [
    "Use web_search to find today's news in Karachi, then summarise it.",
    "Call send_email to tell Ali the meeting moved to 5pm.",
    "What's the weather in Lahore? Use get_weather_now.",
    "Add 'twelve' and 30 using the add tool.",
    "Add \"7\" and \"8\" (pass them as strings exactly) with add.",
    "Get weather for city=Karachi but call the argument 'location'.",
    "Look up the population of Pakistan with slow_lookup.",
    "Fetch user 42 with flaky_db and tell me their name.",
    "Check the weather in Karachi. Then check it again to be sure. Then once more.",
    "Keep calling get_weather for Karachi until it says it's raining.",
    "Save a note, then weather in Quetta, then add 2.5 and 4, then save another note.",
    "Using the calendar tool, book me a slot tomorrow.",
    "Respond ONLY with a tool call that has no name.",
    "Weather in Islamabad, Peshawar, Multan, Sukkur, Gilgit, Hunza, Skardu, Murree, Swat, Chitral and Gwadar.",
    "Hi! What's 2+2? (no tools needed)"
]

def run_stress_test():
    results = []
    
    print(f"Running stress test on {len(PROMPTS)} prompts...\n")
    
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"Testing Prompt #{i}...", end=" ", flush=True)
        
        # --- 1. Run Raw Agent ---
        raw_success = False
        try:
            # Mocking the raw_run to accept a key for speedrun
            import raw_agent
            # We modify the raw_run slightly to use our hardcoded key
            def wrapped_raw(p):
                # This is a quick hack to ensure raw_agent uses the same key
                import os
                os.environ["GEMINI_API_KEY"] = MY_KEY
                return raw_agent.raw_run(p, my_tools)
            
            raw_out = wrapped_raw(prompt)
            if raw_out and len(raw_out.strip()) > 0:
                raw_success = True
        except Exception:
            raw_out = "CRASHED"
            raw_success = False

        # --- 2. Run Harness ---
        harness_success = False
        try:
            h = Client(api_key=MY_KEY)
            h_res = h.run(prompt, tools=my_tools)
            if h_res["text"] and len(h_res["text"].strip()) > 0:
                harness_success = True
        except Exception:
            h_res = "CRASHED"
            harness_success = False

        results.append({
            "id": i,
            "raw": "✅" if raw_success else "❌",
            "harness": "✅" if harness_success else "❌"
        })
        print(f"Raw: {results[-1]['raw']} | Harness: {results[-1]['harness']}")

    # Print Final Table
    print("\n" + "="*30)
    print(f"{'ID':<5} | {'Raw SDK':<10} | {'Harness':<10}")
    print("-" * 30)
    raw_fails = 0
    h_fails = 0
    for r in results:
        print(f"{r['id']:<5} | {r['raw']:<10} | {r['harness']:<10}")
        if r['raw'] == "❌": raw_fails += 1
        if r['harness'] == "❌": h_fails += 1
    
    print("="*30)
    print(f"TOTAL SILENT FAILURES: Raw={raw_fails}/15, Harness={h_fails}/15")
    print("="*30)

if __name__ == "__main__":
    run_stress_test()