import os
from google import genai
from google.genai import types
from tools import get_weather, add, save_note, slow_lookup, flaky_db
from dotenv import load_dotenv # 1. Import the helper

load_dotenv() # 2. This reads the .env file and loads the keys into memory


def raw_run(prompt, tools_list, model="gemma-4-26b-a4b-it", max_steps=5):
    client = genai.Client(MY_KEY = os.environ.get("GEMINI_API_KEY"))
    tool_map = {f.__name__: f for f in tools_list}
    decls = [types.FunctionDeclaration.from_callable(client=client, callable=f) for f in tools_list]
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=decls)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    for step in range(max_steps):
        response = client.models.generate_content(model=model, contents=contents, config=config)
        if not response.function_calls:
            return response.text or ""
        contents.append(response.candidates[0].content)
        tool_responses = []
        for fc in response.function_calls:
            print(f"Step {step}: Calling {fc.name} with {fc.args}")
            result = tool_map[fc.name](**fc.args)
            tool_responses.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=tool_responses))
    return "Max steps reached."

if __name__ == "__main__":
    my_tools = [get_weather, add, save_note, slow_lookup, flaky_db]
    print(f"Result 1: {raw_run('What is 2+2?', my_tools)}")
    print("\n" + "="*30 + "\n")
    print(f"Result 2: {raw_run('Use get_weather_now to find weather in Karachi', my_tools)}")