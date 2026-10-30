"""Tools for the AI agent.

They only read one session's data and return plain dictionaries, so they work
with any LLM (Claude, Gemini, OpenAI). The agent decides which tool to call;
our code runs the tool and sends the result back to the model.
"""

import datetime
import os

from analytics import clean_segments, emotion_segments, emotion_summary

# Per-class F1 of our model on the FER2013 test set (from evaluate.py).
# The agent uses it to know which emotions the model is weak at.
MODEL_TEST_F1 = {
    "angry": 0.46, "disgust": 0.40, "fear": 0.35, "happy": 0.78,
    "neutral": 0.51, "sad": 0.44, "surprise": 0.71,
}

# Recording problems the quality check can detect, with advice for each
FLAG_INFO = {
    "short_session": (
        "The session is very short (under 20 seconds).",
        "Record at least 30 seconds so there is enough data."),
    "face_often_missing": (
        "The face was not detected for a large part of the session.",
        "Stay inside the frame, face the camera and keep your face uncovered."),
    "low_confidence": (
        "Many predictions have low confidence.",
        "Use bright light from the front, keep the camera at eye level and "
        "face it straight."),
    "almost_only_neutral": (
        "Almost every prediction is neutral.",
        "To test the model, hold an exaggerated expression for 3-4 seconds, "
        "with a neutral face in between."),
    "few_photos": (
        "There are very few photos in this session (under 5).",
        "Take more photos before drawing any conclusions."),
}

# Descriptions the LLM sees. Good descriptions = the model picks the right tool.
TOOL_SCHEMAS = [
    {
        "name": "get_session_overview",
        "description": "Basic facts about the session: duration, number of "
                       "predictions, average confidence and how many "
                       "predictions had low confidence. Call this first.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_emotion_summary",
        "description": "Percentage and count of each predicted emotion. "
                       "Optionally ignore predictions below a confidence level.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_confidence": {
                    "type": "number",
                    "description": "Ignore predictions below this confidence "
                                   "(0 to 1). Default 0.",
                },
            },
        },
    },
    {
        "name": "get_segments",
        "description": "How the emotion changed over time, as segments with "
                       "start, end, duration and mean confidence. Very short "
                       "segments are removed as noise. For photo sessions it "
                       "lists the individual photos instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_duration": {
                    "type": "number",
                    "description": "Ignore segments shorter than this many "
                                   "seconds. Default 0.5.",
                },
            },
        },
    },
    {
        "name": "get_emotion_changes",
        "description": "List of moments where the dominant emotion changed "
                       "(from, to, time in seconds). Only meaningful for "
                       "continuous webcam sessions.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_duration": {"type": "number",
                                 "description": "Default 0.5 seconds."},
            },
        },
    },
    {
        "name": "get_confidence_report",
        "description": "Average confidence and number of predictions for each "
                       "emotion, and whether each emotion has enough "
                       "predictions to be reliable. Use it before making "
                       "claims about a specific emotion.",
        "parameters": {"type": "object", "properties": {}},
    },

    {
        "name": "get_recording_quality",
        "description": "Checks how good the RECORDING was (not the person): "
                       "how much of the session had a detected face, how many "
                       "predictions had low confidence, session length. "
                       "Returns a list of issues, each with a tip. Use it for "
                       "questions about recording quality or how to get "
                       "better results.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "save_report",
        "description": "Save the final written report as a Markdown file. "
                       "Call this once at the end, only if the user asked "
                       "for a saved report.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string",
                         "description": "Full report text in Markdown."},
            },
            "required": ["text"],
        },
    },
]

TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}


class SessionTools:
    """All tools for one session. df must have: seconds, emotion, confidence."""

    def __init__(self, df, session_type="webcam"):
        self.df = df
        self.session_type = session_type   # "webcam" or "photos"

    # ---------- tools ----------
    def get_session_overview(self):
        df = self.df
        return {
            "session_type": self.session_type,
            "duration_seconds": round(float(df["seconds"].iloc[-1]), 1),
            "num_predictions": int(len(df)),
            "average_confidence": round(float(df["confidence"].mean()), 2),
            "percent_below_0_5_confidence":
                round(float((df["confidence"] < 0.5).mean() * 100), 1),
        }

    def get_emotion_summary(self, min_confidence=0.0):
        d = self.df[self.df["confidence"] >= min_confidence]
        if d.empty:
            return {"error": "No predictions above this confidence."}
        counts = d["emotion"].value_counts()
        return {
            "percent": emotion_summary(d),
            "counts": {k: int(v) for k, v in counts.items()},
            "predictions_used": int(len(d)),
        }

    def get_segments(self, min_duration=0.5):
        if self.session_type == "photos":
            rows = self.df[["seconds", "emotion", "confidence"]].round(2)
            return {
                "note": "Photo session: each row is one photo, not a "
                        "continuous segment.",
                "photos": rows.to_dict("records"),
            }
        segs = clean_segments(emotion_segments(self.df), min_duration)
        return {"segments": segs.to_dict("records")}

    def get_emotion_changes(self, min_duration=0.5):
        if self.session_type == "photos":
            return {"note": "Not meaningful for photo sessions.",
                    "changes": []}
        segs = clean_segments(emotion_segments(self.df), min_duration)
        rows = segs.to_dict("records")
        changes = [{"at_seconds": b["start"], "from": a["emotion"],
                    "to": b["emotion"]} for a, b in zip(rows, rows[1:])]
        return {"changes": changes}

    def get_confidence_report(self):
        stats = self.df.groupby("emotion")["confidence"].agg(["mean", "count"])
        per_emotion = {}
        for e, r in stats.iterrows():
            enough = bool(r["count"] >= 5)
            confident = bool(r["mean"] >= 0.45)
            per_emotion[e] = {
                "mean_confidence": round(float(r["mean"]), 2),
                "n": int(r["count"]),
                "enough_predictions": enough,
                "low_confidence": not confident,
                "reliable": enough and confident,
                "model_test_f1": MODEL_TEST_F1.get(e),
            }
        return {
            "per_emotion": per_emotion,
            "note": "reliable = at least 5 predictions AND mean confidence "
                    ">= 0.45. These are rough rules of thumb, not statistical "
                    "tests. model_test_f1 shows how well the model recognises "
                    "that emotion on the test set (low = often wrong).",
        }

    def get_recording_quality(self):
        df = self.df
        n = len(df)
        duration = float(df["seconds"].iloc[-1])
        shares = df["emotion"].value_counts(normalize=True)
        avg_conf = float(df["confidence"].mean())
        low_share = float((df["confidence"] < 0.5).mean() * 100)

        flags = []
        result = {
            "session_type": self.session_type,
            "duration_seconds": round(duration, 1),
            "num_predictions": int(n),
            "average_confidence": round(avg_conf, 2),
            "percent_below_0_5_confidence": round(low_share, 1),
        }

        if self.session_type == "photos":
            if n < 5:
                flags.append("few_photos")
        else:
            # A gap = more than 1 second without any prediction (no face found)
            gaps = df["seconds"].diff().dropna()
            gaps = gaps[gaps > 1.0]
            missing = float(gaps.sum())
            coverage = 100.0 * (1 - missing / duration) if duration > 0 else 100.0
            result["face_detected_percent"] = round(coverage, 1)
            result["longest_gap_seconds"] = (
                round(float(gaps.max()), 1) if len(gaps) else 0.0)
            if duration < 20:
                flags.append("short_session")
            if coverage < 80:
                flags.append("face_often_missing")

        if avg_conf < 0.5 or low_share > 40:
            flags.append("low_confidence")
        if n >= 5 and shares.index[0] == "neutral" and shares.iloc[0] > 0.9:
            flags.append("almost_only_neutral")

        result["issues"] = [
            {"flag": f, "meaning": FLAG_INFO[f][0], "tip": FLAG_INFO[f][1]}
            for f in flags]
        result["note"] = (
            "Rule-of-thumb checks, not statistical tests. Time before the "
            "first and after the last detected face is not visible in the log.")
        return result

    def save_report(self, text):
        os.makedirs("reports", exist_ok=True)
        name = datetime.datetime.now().strftime("reports/report_%Y%m%d_%H%M%S.md")
        with open(name, "w", encoding="utf-8") as f:
            f.write(text)
        return {"saved_to": name}

    # ---------- dispatcher ----------
    def call(self, name, arguments=None):
        """Run a tool by name. Errors are returned as data, never raised."""
        if name not in TOOL_NAMES:
            return {"error": f"Unknown tool: {name}"}
        try:
            return getattr(self, name)(**(arguments or {}))
        except Exception as e:   # the model can read the error and recover
            return {"error": str(e)}