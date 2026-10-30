"""Run the agent loop on the latest session.

    python test_agent.py            # scripted stand-in LLM (no API key needed)
    python test_agent.py gemini     # real Gemini model
"""

import glob
import os
import sys

from agent import Agent, ScriptedLLM
from analytics import load_log
from tools import SessionTools

paths = sorted(glob.glob("sessions/*.csv"))
if not paths and os.path.exists("session_log.csv"):
    paths = ["session_log.csv"]
if not paths:
    raise SystemExit("No session CSV found. Run webcam_app.py first.")

if len(sys.argv) > 1 and sys.argv[1] == "gemini":
    from llm_gemini import GeminiLLM
    llm = GeminiLLM()
    print("LLM: Gemini,", llm.model)
else:
    llm = ScriptedLLM()
    print("LLM: scripted stand-in")

agent = Agent(llm, SessionTools(load_log(paths[-1])))
question = sys.argv[2] if len(sys.argv) > 2 else "Analyse this session and write a short report."
text, trace = agent.run(question)

print("\nTool calls made:")
for name, args, _ in trace:
    print("  -", name, args)
print("\n" + text)