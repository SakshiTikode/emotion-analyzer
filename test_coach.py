"""Check the recording quality of the latest session.

    python test_coach.py           # quality check only (no API key needed)
    python test_coach.py gemini    # plus coaching tips from the agent
"""

import glob
import json
import os
import sys

from dotenv import load_dotenv

from agent import COACH_PROMPT, Agent
from analytics import load_log
from tools import SessionTools

load_dotenv(override=True)

paths = sorted(glob.glob("sessions/*.csv"))
if not paths and os.path.exists("session_log.csv"):
    paths = ["session_log.csv"]
if not paths:
    raise SystemExit("No session CSV found. Run webcam_app.py first.")

tools = SessionTools(load_log(paths[-1]))
print("Using:", paths[-1])
print(json.dumps(tools.call("get_recording_quality"), indent=2))

if len(sys.argv) > 1 and sys.argv[1] == "gemini":
    from llm_gemini import GeminiLLM
    agent = Agent(GeminiLLM(), tools, system_prompt=COACH_PROMPT)
    text, trace = agent.run("How good was this recording? How can I improve it?")
    print("\nTools used:", [name for name, _, _ in trace])
    print("\n" + text)