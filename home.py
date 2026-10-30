"""Home page: project overview with interactive sections."""

import plotly.graph_objects as go
import streamlit as st

from common import CLASS_NAMES, EMOJI

# Test-set results on FER2013 (7,178 images), copied from our evaluate.py output
RESULTS = {
    "Baseline CNN": {
        "accuracy": 0.539, "macro_f1": 0.435,
        "f1": {"angry": 0.397, "disgust": 0.000, "fear": 0.222,
               "happy": 0.814, "neutral": 0.544, "sad": 0.369,
               "surprise": 0.700},
    },
    "MobileNetV2 (transfer learning)": {
        "accuracy": 0.561, "macro_f1": 0.523,
        "f1": {"angry": 0.464, "disgust": 0.398, "fear": 0.352,
               "happy": 0.782, "neutral": 0.512, "sad": 0.435,
               "surprise": 0.714},
    },
}

# MobileNetV2 per emotion: (precision, recall, F1, number of test images)
DETAILS = {
    "angry": (0.408, 0.539, 0.464, 958),
    "disgust": (0.420, 0.378, 0.398, 111),
    "fear": (0.471, 0.281, 0.352, 1024),
    "happy": (0.770, 0.795, 0.782, 1774),
    "neutral": (0.477, 0.553, 0.512, 1233),
    "sad": (0.439, 0.431, 0.435, 1247),
    "surprise": (0.770, 0.665, 0.714, 831),
}

PIPELINE = [
    ("1. Camera",
     "The browser camera takes a photo. (Locally, webcam_app.py reads live "
     "video from your laptop camera instead.)"),
    ("2. Face detection",
     "A Haar Cascade detector finds the largest face and draws the green box."),
    ("3. Preprocess",
     "The face is cropped, converted to grayscale and resized to 48x48 "
     "pixels, the same format as the FER2013 training images."),
    ("4. Model",
     "A fine-tuned MobileNetV2 outputs 7 probabilities, one per emotion."),
    ("5. Result",
     "The highest probability is the predicted emotion and its value is the "
     "confidence. Neither one is a fact about how the person feels."),
]

# ---------------- Hero ----------------
st.title("🎭 Emotion Analyzer")
st.write("Estimate emotions from facial expressions with a CNN trained on "
         "FER2013, then explore the results.")

b1, b2, _ = st.columns([1, 1, 3])
with b1:
    st.page_link("views/1_Try_it.py", label="Take a photo", icon="📸")
with b2:
    st.page_link("views/2_Session_Report.py", label="Session report", icon="📊")

log = st.session_state.get("log", [])
if log:
    last = log[-1][1]
    st.success(f"{len(log)} photo(s) analysed in this session. "
               f"Last result: {EMOJI[last]} {last}")
else:
    st.info("No photos yet. Start with **Try it**.")

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["How it works", "Model results", "Emotion explorer", "Limits & privacy"])

with tab1:
    st.subheader("From camera to emotion")
    stage = st.radio("Pipeline stage", [p[0] for p in PIPELINE],
                     horizontal=True, label_visibility="collapsed")
    st.info(dict(PIPELINE)[stage])

with tab2:
    st.subheader("How well does the model do?")
    chosen = st.multiselect("Models to show", list(RESULTS),
                            default=list(RESULTS))
    if not chosen:
        st.info("Pick at least one model.")
    else:
        cols = st.columns(len(chosen))
        for col, name in zip(cols, chosen):
            col.markdown(f"**{name}**")
            col.metric("Accuracy", f"{RESULTS[name]['accuracy'] * 100:.1f}%")
            col.metric("Macro F1", f"{RESULTS[name]['macro_f1']:.3f}")

        fig = go.Figure()
        for name in chosen:
            fig.add_trace(go.Bar(
                name=name,
                x=[f"{EMOJI[e]} {e}" for e in CLASS_NAMES],
                y=[RESULTS[name]["f1"][e] for e in CLASS_NAMES],
            ))
        fig.update_layout(barmode="group", height=380,
                          yaxis=dict(range=[0, 1], title="F1 score"),
                          margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig)
        st.caption("Macro F1 averages the F1 of all 7 emotions equally, so a "
                   "rare emotion like disgust counts as much as happy.")

with tab3:
    st.subheader("How reliable is each emotion?")
    emotion = st.selectbox("Pick an emotion", CLASS_NAMES,
                           format_func=lambda e: f"{EMOJI[e]} {e}")
    p, r, f1, n = DETAILS[emotion]

    a, b, c, d = st.columns(4)
    a.metric("Precision", f"{p:.2f}")
    b.metric("Recall", f"{r:.2f}")
    c.metric("F1", f"{f1:.2f}")
    d.metric("Test images", n)

    level = "strong" if f1 >= 0.6 else "medium" if f1 >= 0.4 else "weak"
    st.markdown(
        f"When the model says **{emotion}**, it is right about "
        f"**{p * 100:.0f} times out of 100** (precision). Of 100 real "
        f"**{emotion}** faces, it finds about **{r * 100:.0f}** (recall). "
        f"So the model is **{level}** on this emotion."
    )
    st.caption("Rough scale used here: F1 of 0.6 or more is strong, 0.4 to "
               "0.6 is medium, below 0.4 is weak. Results are from the "
               "MobileNetV2 model on the FER2013 test set.")

with tab4:
    st.warning(
        "This model estimates emotion from **facial expressions only**, with "
        "about 56% test accuracy. It cannot know what a person is thinking "
        "or feeling."
    )
    with st.expander("Known weaknesses"):
        st.markdown(
            "- Fear, sad and angry are often confused with each other\n"
            "- Disgust has very little training data (436 images vs 7,215 "
            "for happy)\n"
            "- Real webcams differ from FER2013 images, so results can be "
            "worse in practice (dark rooms, odd angles)\n"
            "- A smile or neutral face does not mean the person feels that way"
        )
    with st.expander("Privacy"):
        st.markdown(
            "- Photos are processed in memory and are not saved\n"
            "- Only the predicted emotion is kept, for the current browser "
            "session\n"
            "- Clear it any time with the **Clear session** button in the "
            "sidebar"
        )