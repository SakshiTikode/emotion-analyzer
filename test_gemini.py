"""Check that the Gemini API key works and see which Flash models it can use."""

import os

from dotenv import load_dotenv
from google import genai

# Read GEMINI_API_KEY from the .env file in the project folder.
# override=True means .env wins over a value set earlier in the terminal.
load_dotenv(override=True)

if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit("GEMINI_API_KEY not found. Check the .env file.")

client = genai.Client()   # reads GEMINI_API_KEY from the environment

print("Flash models available to your key:")
for m in client.models.list():
    if "flash" in m.name:
        print("  ", m.name)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
response = client.models.generate_content(
    model=MODEL, contents="Reply with exactly: key works")
print(f"\n{MODEL} says:", response.text)