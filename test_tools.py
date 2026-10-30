"""Run every read-only tool on the latest session and print the results."""

import glob
import json
import os

from analytics import load_log
from tools import SessionTools

paths = sorted(glob.glob("sessions/*.csv"))
if not paths and os.path.exists("session_log.csv"):
    paths = ["session_log.csv"]
if not paths:
    raise SystemExit("No session CSV found. Run webcam_app.py first.")

print("Using:", paths[-1])
tools = SessionTools(load_log(paths[-1]))

for name in ["get_session_overview", "get_emotion_summary", "get_segments",
             "get_emotion_changes", "get_confidence_report"]:
    print(f"\n--- {name} ---")
    print(json.dumps(tools.call(name), indent=2))