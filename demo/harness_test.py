import os
from dotenv import load_dotenv # 1. Import the helper
from tools import get_weather, add, save_note, slow_lookup, flaky_db
from gemma_harness.client import Client

load_dotenv() # 2. This reads the .env file and loads the keys into memory

# Use your actual key here
MY_KEY = os.environ.get("GEMINI_API_KEY")
my_tools = [get_weather, add, save_note, slow_lookup, flaky_db]

def test_harness(prompt):
    print(f"\nPrompt: {prompt}")
    h = Client(api_key=MY_KEY, on_event=lambda ev: print(f"  [TRACE] {ev['label']}"))
    res = h.run(prompt, tools=my_tools)
    print(f"Final Answer: {res['text']}")

if __name__ == "__main__":
    # Test 1: Normal
    test_harness("What is 2+2?")
    # Test 2: Misspelled tool (The "Fix" test)
    test_harness("Use get_weather_now to find weather in Karachi")
    # Test 3: Phantom tool (The "Lint" test)
    test_harness("Use web_search to find the news")
    # Test 4: Tool crash (The "Recovery" test)
    test_harness("Fetch user 42 with flaky_db")