"""Baseline agent loop with NO recovery (the "before" column)."""
from google import genai
from google.genai import types


def raw_run(prompt, tools, model="gemma-4-26b-a4b-it", max_steps=8):
    c = genai.Client()
    tm = {f.__name__: f for f in tools}
    decls = [types.FunctionDeclaration.from_callable(client=c, callable=f) for f in tools]
    cfg = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=decls)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    for _ in range(max_steps):
        r = c.models.generate_content(model=model, contents=contents, config=cfg)
        if not r.function_calls:
            return r.text or ""  # "" = silent failure
        contents.append(r.candidates[0].content)
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=fc.name, response={"result": tm[fc.name](**fc.args)}
                    )
                    for fc in r.function_calls
                ],
            )
        )  # crashes on bad name/args
    return ""
