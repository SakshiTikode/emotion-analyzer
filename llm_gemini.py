"""Gemini adapter for the agent loop.

It translates our neutral message format to the Gemini API and back. Only this
file knows about Gemini; the Agent class does not.
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv(override=True)


class GeminiLLM:
    def __init__(self, model=None, temperature=0.2):
        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit("GEMINI_API_KEY not found. Check the .env file.")
        self.client = genai.Client()   # reads GEMINI_API_KEY from the environment
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.temperature = temperature

    # ---------- our format -> Gemini ----------
    @staticmethod
    def _declarations(tools):
        decls = []
        for t in tools:
            d = {"name": t["name"], "description": t["description"]}
            params = t.get("parameters", {})
            if params.get("properties"):   # Gemini rejects empty parameter objects
                d["parameters"] = params
            decls.append(types.FunctionDeclaration(**d))
        return decls

    @staticmethod
    def _to_contents(messages):
        contents, pending = [], []

        def flush():
            # Results of tools called together must go back in ONE message
            if pending:
                contents.append(types.Content(role="user", parts=list(pending)))
                pending.clear()

        for m in messages:
            if m["role"] == "tool":
                pending.append(types.Part(function_response=types.FunctionResponse(
                    id=m["id"], name=m["name"], response={"result": m["result"]})))
                continue
            flush()
            if m["role"] == "user":
                contents.append(types.Content(
                    role="user", parts=[types.Part.from_text(text=m["content"])]))
            elif m.get("raw") is not None:
                # Send back the model's own reply unchanged (keeps call ids)
                contents.append(m["raw"])
            else:
                contents.append(types.Content(
                    role="model", parts=[types.Part.from_text(text=m["text"] or " ")]))
        flush()
        return contents

    # ---------- the one method the Agent calls ----------
    def generate(self, system, messages, tools):
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=self._declarations(tools))],
            temperature=self.temperature,
        )
        contents = self._to_contents(messages)

        response, last_error = None, None
        for attempt in range(4):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config)
                break
            except (errors.ServerError, errors.ClientError) as e:
                last_error = e
                code = getattr(e, "code", None)
                if code not in (429, 500, 503):
                    raise   # not temporary (bad key, bad request, ...)
                if attempt < 3:
                    wait = 10 * (attempt + 1)   # waits 10, 20, 30 seconds
                    print(f"Gemini returned {code} (temporary). Retrying in {wait} s...")
                    time.sleep(wait)
        if response is None:
            raise last_error

        # ---------- Gemini -> our format ----------
        candidates = response.candidates or []
        if not candidates or candidates[0].content is None:
            return {"text": "The model returned no content.",
                    "tool_calls": [], "raw": None}

        content = candidates[0].content
        text, calls = "", []
        for part in content.parts or []:
            if part.function_call:
                fc = part.function_call
                calls.append({"id": fc.id, "name": fc.name,
                              "arguments": dict(fc.args or {})})
            elif part.text and not getattr(part, "thought", False):
                text += part.text
        return {"text": text, "tool_calls": calls, "raw": content}