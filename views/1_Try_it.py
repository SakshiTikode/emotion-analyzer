"""Take a photo with the browser camera and predict the emotion.

Two modes:
  Predict:  the result is added to the current session (Session Report page).
  Practice: pick a target emotion, see how well the model recognises it,
            and get tips. Practice photos are not added to the session.
"""

import hashlib

import cv2
import numpy as np
import streamlit as st

from common import (CLASS_NAMES, EMOJI, load_detector, load_model,
                    log_photo_result)
from practice import (PRACTICE_PROMPT, PRACTICE_TIPS, PracticeTools,
                      confusion_tip)
from tools import MODEL_TEST_F1

model = load_model()
detector = load_detector()

st.title("📸 Try it")
mode = st.radio("Mode", ["Predict", "Practice"], horizontal=True)

if mode == "Practice":
    target = st.selectbox("Emotion to practice", CLASS_NAMES,
                          format_func=lambda e: f"{EMOJI[e]} {e}")
    with st.expander("How to make this expression"):
        for tip in PRACTICE_TIPS[target]:
            st.write("- " + tip)
    st.info("Practice photos are processed in memory and are not added to "
            "the session report. Results are rough estimates.")
else:
    st.info("Your photo is processed in memory and is not saved. Only the "
            "predicted emotion is kept for this browser session. Results are "
            "rough estimates (about 56% model accuracy).")

photo = st.camera_input("Take a photo", key=f"camera_{mode}")

if photo is not None:
    # Decode the photo sent by the browser
    data = np.frombuffer(photo.getvalue(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        st.warning("No face detected. Try better lighting and face the camera.")
    else:
        # Use the largest face and prepare it the way the model expects
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = cv2.resize(gray[y:y + h, x:x + w], (48, 48),
                          interpolation=cv2.INTER_AREA)
        inp = face.astype("float32").reshape(1, 48, 48, 1)
        probs = model(inp, training=False).numpy()[0]

        top = int(np.argmax(probs))
        name = CLASS_NAMES[top]

        if mode == "Predict":
            # Remember this result for the Session Report page
            if log_photo_result(photo.getvalue(), name, probs[top]):
                st.rerun()   # refresh the sidebar counter

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        st.image(frame, channels="BGR")

        if mode == "Predict":
            st.subheader(f"{EMOJI[name]} {name} ({probs[top] * 100:.0f}%)")
            st.caption(f"Photos in this session: "
                       f"{len(st.session_state['log'])}. Open the Session "
                       "Report page to see them together.")
        else:
            score = float(probs[CLASS_NAMES.index(target)])
            st.metric(f"Match with {EMOJI[target]} {target}",
                      f"{score * 100:.0f}%")

            if name == target:
                st.success(f"The model recognised {target}.")
            else:
                st.warning(f"The model saw {EMOJI[name]} {name} "
                           f"({probs[top] * 100:.0f}%) instead.")
                st.info(confusion_tip(target, name))

            f1 = MODEL_TEST_F1.get(target)
            if f1 is not None and f1 < 0.45:
                st.caption(f"Note: this model recognises {target} poorly "
                           f"(test F1 {f1:.2f}), so a low match may be the "
                           "model's weakness, not yours.")

            # Optional: the agent writes a personal tip (needs the Gemini key)
            photo_id = hashlib.md5(photo.getvalue()).hexdigest()
            if st.button("🤖 Get a personal tip from the agent",
                         key="practice_agent"):
                with st.spinner("The agent is looking at your attempt..."):
                    try:
                        from agent import Agent
                        from llm_gemini import GeminiLLM

                        tools = PracticeTools(target, dict(
                            zip(CLASS_NAMES, map(float, probs))))
                        coach = Agent(GeminiLLM(), tools,
                                      system_prompt=PRACTICE_PROMPT)
                        tip, _ = coach.run("Give me feedback on this attempt.")
                        st.session_state["practice_tip"] = (photo_id, tip)
                    except SystemExit as e:      # for example: API key missing
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"The AI agent could not run: {e}")

            saved = st.session_state.get("practice_tip")
            if saved and saved[0] == photo_id:
                st.markdown(saved[1])

        with st.expander("All probabilities", expanded=(mode == "Predict")):
            for n, p in zip(CLASS_NAMES, probs):
                st.write(f"{EMOJI[n]} {n}: {p * 100:.1f}%")
                st.progress(float(p))