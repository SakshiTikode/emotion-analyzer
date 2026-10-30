"""Entry point: navigation and the sidebar shown on every page.

Run the whole app with:  streamlit run app.py
"""

import streamlit as st

from common import EMOJI

st.set_page_config(page_title="Emotion Analyzer", page_icon="🎭", layout="wide")

# ---------------- Navigation ----------------
home = st.Page("home.py", title="Home", icon="🏠", default=True)
try_it = st.Page("views/1_Try_it.py", title="Try it", icon="📸")
report = st.Page("views/2_Session_Report.py", title="Session Report", icon="📊")

pg = st.navigation({"Explore": [home], "Analyze": [try_it, report]})

# ---------------- Sidebar shown on every page ----------------
with st.sidebar:
    st.divider()
    st.subheader("🧾 This session")

    log = st.session_state.get("log", [])
    if log:
        last_emotion = log[-1][1]
        c1, c2 = st.columns(2)
        c1.metric("Photos", len(log))
        c2.metric("Last", f"{EMOJI[last_emotion]}")
        st.caption(f"Last result: **{last_emotion}**")

        st.page_link(report, label="Open the report", icon="📊")
        if st.button("🗑️ Clear session", key="clear_session"):
            st.session_state["log"] = []
            st.rerun()
    else:
        st.caption("No photos yet. Open **Try it** and take one.")

    with st.expander("ℹ️ About the model"):
        st.markdown(
            "- CNN (MobileNetV2, fine-tuned) trained on FER2013\n"
            "- Test accuracy: **56.1%**, macro F1: **0.52**\n"
            "- It estimates emotion from **facial expressions only**; "
            "it cannot know what someone feels or thinks\n"
            "- Fear, sad and angry are often confused\n"
            "- Photos are processed in memory and not stored"
        )

pg.run()