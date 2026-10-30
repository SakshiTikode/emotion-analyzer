"""Provider-independent AI agent loop.

The agent = an LLM + tools + this loop. The LLM decides which tool to call,
our code runs the tool and gives the result back, and this repeats until the
LLM writes the final answer.
"""

SYSTEM_PROMPT = """You analyse one recorded emotion session.

The data comes from a CNN that predicts emotions from facial expressions. It
has about 56% accuracy on its test set, so its predictions are rough estimates.

Rules:
1. Start by calling get_session_overview and get_confidence_report. Base every
   statement on numbers returned by the tools. Never invent numbers.
2. Say "the model predicted ...", never "the person felt ...". A facial
   expression is not the same as a feeling.
3. Do not draw conclusions about emotions marked reliable=false. You may
   mention them, but say clearly that they are uncertain.
4. Use model_test_f1 from the confidence report: emotions with a low value
   (for example fear) are often predicted wrongly by this model.
5. "neutral" is the model's default prediction. A large neutral share is not
   evidence that the person showed no emotion.
6. Treat very short segments (about 1 second or less) as noise.
7. Make no medical, psychological or personality claims, and no judgements
   about honesty, ability or mental health.
8. If the user asks whether the person was lying, nervous, anxious, stressed,
   depressed or similar, say in the first sentence that this cannot be
   determined from these predictions. Then give at most 3 short bullets with
   the facts that are relevant to the question. Do not agree with a premise
   in the question just because it is phrased as a fact.
9. If the user asks for a general analysis or report, use these headings:
   Summary, Timeline, Reliability notes, Limitations. Keep it under 250 words
   and mention only the important changes in the Timeline, not every segment.
10. Put every bullet point on its own line.
"""

COACH_PROMPT = """You are a recording coach for an emotion-recognition app.
You judge the quality of the RECORDING, never the person.

Rules:
1. Call get_recording_quality first. Base everything on its numbers and issues.
2. If there are no issues, say the recording looks fine and stop.
3. Otherwise explain the most important problems in plain words and give at
   most 4 concrete tips. Use the tips from the tool and explain why each helps.
4. Do not say anything about the person's feelings, honesty or personality.
5. Keep it under 120 words. Put every bullet on its own line.
"""

class Agent:
    def __init__(self, llm, tools, max_steps=6, system_prompt=SYSTEM_PROMPT):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps   # safety limit so the loop can never run forever
        self.system_prompt = system_prompt

    def run(self, user_message):
        """Returns (final_text, trace). trace lists every tool call that was made."""
        messages = [{"role": "user", "content": user_message}]
        trace = []

        for _ in range(self.max_steps):
            reply = self.llm.generate(self.system_prompt, messages, self.tools_schemas())
            messages.append({"role": "assistant", "text": reply["text"],
                             "tool_calls": reply["tool_calls"],
                             "raw": reply.get("raw")})   # provider's own reply object

            if not reply["tool_calls"]:          # no tool requested = final answer
                return reply["text"], trace

            for call in reply["tool_calls"]:     # run every requested tool
                result = self.tools.call(call["name"], call["arguments"])
                trace.append((call["name"], call["arguments"], result))
                messages.append({"role": "tool", "id": call["id"],
                                 "name": call["name"], "result": result})

        return "Stopped: the agent used too many steps.", trace

    
    def tools_schemas(self):
        from tools import TOOL_SCHEMAS
        return getattr(self.tools, "schemas", TOOL_SCHEMAS)


class ScriptedLLM:
    """A stand-in for a real LLM, used to test the loop without an API key.

    It is NOT a model: it always asks for the same tools and then fills a
    template from their results. A real LLM replaces this class later.
    """

    def generate(self, system, messages, tools):
        results = {m["name"]: m["result"] for m in messages if m["role"] == "tool"}

        if not results:   # first turn: ask for the tools
            names = ["get_session_overview", "get_emotion_summary",
                     "get_confidence_report", "get_emotion_changes"]
            return {"text": "", "tool_calls": [
                {"id": str(i), "name": n, "arguments": {}}
                for i, n in enumerate(names)]}

        # second turn: write the report from the tool results
        ov = results["get_session_overview"]
        summ = results["get_emotion_summary"]["percent"]
        conf = results["get_confidence_report"]["per_emotion"]
        changes = results["get_emotion_changes"]["changes"]
        top = max(summ, key=summ.get)

        trusted = [e for e, v in conf.items() if v["reliable"]]
        shaky = [e for e, v in conf.items() if not v["reliable"]]

        lines = [
            f"**Summary:** {ov['duration_seconds']} s session with "
            f"{ov['num_predictions']} predictions. The model predicted "
            f"**{top}** most often ({summ[top]}%).",
        ]
        if changes:
            moves = "; ".join(f"{c['from']} -> {c['to']} at {c['at_seconds']} s"
                              for c in changes)
            lines.append(f"**Timeline:** {moves}.")
        lines.append("**Reliability notes:** reliable predictions: "
                     + (", ".join(trusted) or "none") + ".")
        if shaky:
            lines.append("Treat with caution (low confidence or few "
                         "predictions): " + ", ".join(shaky) + ".")
        lines.append(
            f"**Limitations:** {ov['percent_below_0_5_confidence']}% of "
            "predictions had confidence below 0.5. The model has about 56% "
            "test accuracy and reads facial expressions, not feelings.")
        return {"text": "\n\n".join(lines), "tool_calls": []}