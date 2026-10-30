"""Expression practice coach: tips and agent tools for the practice mode."""

from tools import MODEL_TEST_F1

PRACTICE_TIPS = {
    "angry": ["Pull the eyebrows down and together.",
              "Narrow the eyes and glare.",
              "Press the lips together tightly."],
    "disgust": ["Wrinkle the nose.", "Raise the upper lip.",
                "Squint slightly."],
    "fear": ["Open the eyes wide.",
             "Raise the eyebrows and pull them together.",
             "Open the mouth slightly and stretch the lips sideways."],
    "happy": ["Smile widely so the cheeks rise.",
              "Let the eyes crinkle slightly."],
    "neutral": ["Relax the whole face.",
                "Keep the mouth closed and the eyebrows level."],
    "sad": ["Raise only the inner ends of the eyebrows.",
            "Let the upper eyelids droop slightly.",
            "Pull the mouth corners down."],
    "surprise": ["Open the eyes wide.", "Raise the eyebrows high.",
                 "Open the mouth in a round shape."],
}

# (target emotion, what the model saw instead) -> how to fix it
CONFUSION_TIPS = {
    ("angry", "sad"): "Sad lifts the inner eyebrows. For angry, push the "
                      "eyebrows down and together and keep the mouth tight, "
                      "not turned down.",
    ("angry", "neutral"): "Exaggerate more: lower the eyebrows further and "
                          "glare.",
    ("sad", "neutral"): "Pull the mouth corners down more and raise the "
                        "inner ends of the eyebrows.",
    ("sad", "angry"): "Relax the brow. Raise the inner ends of the eyebrows "
                      "instead of lowering them.",
    ("fear", "surprise"): "Fear is tense: pull the eyebrows together and "
                          "stretch the lips sideways instead of a round "
                          "open mouth.",
    ("fear", "neutral"): "Open the eyes wider and raise the eyebrows more.",
    ("disgust", "angry"): "Focus on the nose and upper lip: wrinkle the nose "
                          "and lift the upper lip instead of lowering the "
                          "eyebrows.",
    ("surprise", "fear"): "Keep the eyebrows raised but relaxed, with a round "
                          "open mouth.",
    ("happy", "neutral"): "Smile wider so the cheeks lift.",
}
GENERIC_TIP = ("Hold a stronger version of the expression for 3-4 seconds, "
               "with good light from the front.")


def confusion_tip(target, seen):
    return CONFUSION_TIPS.get((target, seen), GENERIC_TIP)


PRACTICE_PROMPT = """You are a friendly coach for practising facial expressions
in a demo app. You talk about the FACIAL EXPRESSION only.

Rules:
1. Call get_attempt_result first, then get_tips. Call get_model_reliability
   if the match is low.
2. Give at most 3 concrete tips, based on the tips returned by the tools.
3. If the model's test F1 for the target emotion is below 0.45, say that a low
   match may be the model's weakness and not the user's.
4. Never say anything about the person's real feelings, character or health.
5. Keep it under 100 words. Put every bullet on its own line.
"""

SCHEMAS = [
    {"name": "get_attempt_result",
     "description": "Result of the practice attempt: the target emotion, what "
                    "the model saw, the match percentage and all "
                    "probabilities.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_tips",
     "description": "How to make the target expression, plus a fix for what "
                    "the model saw instead.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_model_reliability",
     "description": "How well the model recognises the target emotion on its "
                    "test set (F1 score).",
     "parameters": {"type": "object", "properties": {}}},
]


class PracticeTools:
    """Tools for one practice attempt. probs = {emotion: probability 0-1}."""

    def __init__(self, target, probs):
        self.target = target
        self.probs = probs
        self.schemas = SCHEMAS    # the Agent reads this attribute

    def _seen(self):
        return max(self.probs, key=self.probs.get)

    def get_attempt_result(self):
        ranked = sorted(self.probs, key=self.probs.get, reverse=True)
        seen = ranked[0]
        return {
            "target_emotion": self.target,
            "model_saw": seen,
            "target_match_percent": round(self.probs[self.target] * 100, 1),
            "model_saw_percent": round(self.probs[seen] * 100, 1),
            "target_rank_among_7": ranked.index(self.target) + 1,
            "all_percent": {e: round(p * 100, 1)
                            for e, p in self.probs.items()},
        }

    def get_tips(self):
        result = {"basic_tips": PRACTICE_TIPS[self.target]}
        seen = self._seen()
        if seen != self.target:
            result["fix_for_what_the_model_saw"] = confusion_tip(
                self.target, seen)
        return result

    def get_model_reliability(self):
        return {
            "emotion": self.target,
            "model_test_f1": MODEL_TEST_F1.get(self.target),
            "note": "F1 below 0.45 means the model often misjudges this "
                    "emotion, so a low match may be the model's weakness.",
        }

    def call(self, name, arguments=None):
        if name not in {s["name"] for s in SCHEMAS}:
            return {"error": f"Unknown tool: {name}"}
        try:
            return getattr(self, name)(**(arguments or {}))
        except Exception as e:
            return {"error": str(e)}