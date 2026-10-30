"""Session report: distribution, timeline and segments.

Sources: the current photo session, recorded webcam sessions (sessions/*.csv),
or an uploaded CSV.
"""

import glob
import os

import plotly.graph_objects as go
import streamlit as st

from analytics import (EMOTIONS, clean_segments, df_from_rows,
                       emotion_segments, emotion_summary, load_log)
from common import COLORS, EMOJI


# Rounded "card" look for the metric boxes
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: #1A1F2B;
        border: 1px solid #2A3042;
        border-radius: 14px;
        padding: 16px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Session Report")
st.caption(
    "Emotions are estimated from facial expressions only. The model has about "
    "56% test accuracy, so treat the results as rough estimates, not facts "
    "about how someone feels."
)

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")

if st.sidebar.button("🗑️ Clear photo session"):
    st.session_state["log"] = []
    st.rerun()

photo_rows = st.session_state.get("log", [])

options = []
if photo_rows:
    options.append("Current session (photos)")
options += sorted(glob.glob("sessions/*.csv"), reverse=True)   # newest first
if os.path.exists("session_log.csv"):
    options.append("session_log.csv")

chosen = st.sidebar.selectbox("Session", options) if options else None
uploaded = st.sidebar.file_uploader("...or upload a CSV", type="csv")
min_conf = st.sidebar.slider("Minimum confidence", 0.0, 0.9, 0.0, 0.05)
min_dur = st.sidebar.slider("Ignore segments shorter than (seconds)",
                            0.0, 3.0, 0.5, 0.1)

# ---------------- Load the data ----------------
photo_mode = False
if uploaded is not None:
    df = load_log(uploaded)
elif chosen == "Current session (photos)":
    df = df_from_rows(photo_rows)
    photo_mode = True
elif chosen is not None:
    df = load_log(chosen)
else:
    st.info("No data yet. Take photos on the **Try it** page, run "
            "`webcam_app.py` locally, or upload a CSV in the sidebar.")
    st.stop()

if df.empty:
    st.warning("The session is empty.")
    st.stop()

# Drop low-confidence predictions (keeps the original 'seconds' values)
df = df[df["confidence"] >= min_conf].reset_index(drop=True)
if df.empty:
    st.warning("No predictions above this confidence. Lower the slider.")
    st.stop()

duration = df["seconds"].iloc[-1]
summary = emotion_summary(df)
top_value = max(summary.values())
top_emotions = [e for e, v in summary.items() if v == top_value]

# ---------------- Key numbers ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("⏱️ Duration", f"{duration:.1f} s")
c2.metric("📷 Photos" if photo_mode else "📊 Predictions", len(df))
c3.metric("🏆 Most frequent",
          " / ".join(f"{EMOJI[e]} {e}" for e in top_emotions))
c4.metric("🎯 Avg confidence", f"{df['confidence'].mean():.2f}")

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Timeline", "Photos" if photo_mode else "Segments",
     "Raw data", "🤖 AI feedback", "💬 Ask the agent"])
with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("Emotion distribution")
        labels = [e for e in EMOTIONS if summary.get(e, 0) > 0]
        pie = go.Figure(go.Pie(
            labels=[f"{EMOJI[e]} {e}" for e in labels],
            values=[summary[e] for e in labels],
            hole=0.55,
            marker=dict(colors=[COLORS[e] for e in labels]),
            textinfo="label+percent",
            sort=False,
        ))
        pie.update_layout(showlegend=False, height=380,
                          margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(pie)

    with right:
        st.subheader("Average confidence per emotion")
        stats = (df.groupby("emotion")["confidence"].agg(["mean", "count"])
                   .reindex(EMOTIONS).dropna())
        bars = go.Figure(go.Bar(
            x=[f"{EMOJI[e]} {e}" for e in stats.index],
            y=stats["mean"],
            marker_color=[COLORS[e] for e in stats.index],
            text=[f"{m:.2f} (n={int(n)})"
                  for m, n in zip(stats["mean"], stats["count"])],
            textposition="outside",
        ))
        bars.update_layout(yaxis=dict(range=[0, 1], title="Confidence"),
                           height=380, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(bars)
        st.caption("n = number of predictions. Emotions with very few "
                   "predictions are not reliable.")

with tab2:
    st.subheader("Timeline")
    x_max = max(duration, 1)

    if photo_mode:
        # Each photo is a single point in time
        fig = go.Figure(go.Scatter(
            x=df["seconds"], y=df["emotion"], mode="markers",
            marker=dict(size=16, color=[COLORS[e] for e in df["emotion"]]),
            customdata=df["confidence"],
            hovertemplate=("%{y}<br>%{x:.0f}s after the first photo<br>"
                           "confidence %{customdata:.2f}<extra></extra>"),
        ))
        fig.update_xaxes(title="Seconds since the first photo",
                         range=[-0.05 * x_max, x_max * 1.05])
        fig.update_yaxes(categoryorder="array", categoryarray=EMOTIONS)
        fig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig)
    else:
        segs = clean_segments(emotion_segments(df), min_duration=min_dur)
        if segs.empty:
            st.info("No segments longer than the selected minimum duration.")
        else:
            fig = go.Figure()
            for _, r in segs.iterrows():
                fig.add_trace(go.Bar(
                    x=[r["duration"]], y=[r["emotion"]], base=[r["start"]],
                    orientation="h",
                    marker_color=COLORS[r["emotion"]],
                    showlegend=False,
                    hovertemplate=(f"{r['emotion']}<br>"
                                   f"{r['start']:.1f}s to {r['end']:.1f}s<br>"
                                   f"confidence {r['mean_conf']:.2f}"
                                   "<extra></extra>"),
                ))
            fig.update_yaxes(categoryorder="array", categoryarray=EMOTIONS)
            fig.update_xaxes(title="Seconds since start", range=[0, x_max])
            fig.update_layout(height=380, bargap=0.3,
                              margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig)
            st.caption("Hover over a block for details. Very short blocks "
                       "are hidden by the slider in the sidebar.")

with tab3:
    if photo_mode:
        st.subheader("Photos in this session")
        shown = df[["seconds", "emotion", "confidence"]].round(2)
        shown["emotion"] = shown["emotion"].map(lambda e: f"{EMOJI[e]} {e}")
        st.dataframe(shown)
    else:
        st.subheader("Segments")
        if segs.empty:
            st.info("No segments to show.")
        else:
            shown = segs.copy()
            shown["emotion"] = shown["emotion"].map(lambda e: f"{EMOJI[e]} {e}")
            st.dataframe(shown)
            st.download_button("⬇️ Download segments (CSV)",
                               segs.to_csv(index=False),
                               file_name="segments.csv", mime="text/csv")

with tab4:
    st.subheader("Raw log (first 200 rows)")
    st.dataframe(df.head(200))

with tab5:
    st.subheader("🤖 AI feedback")
    st.caption(
        "An AI agent reads this session's numbers (never the photos) and "
        "writes a short report. It uses the current sidebar filters. It can "
        "make mistakes, so check it against the charts."
    )

    # One saved report per session, so reruns do not call the API again
    source_name = uploaded.name if uploaded is not None else chosen
    report_key = f"{source_name}|{len(df)}|{min_conf}"

    if st.button("Generate AI feedback"):
        with st.spinner("The agent is analysing the session..."):
            try:
                from agent import Agent
                from llm_gemini import GeminiLLM
                from tools import SessionTools

                tools = SessionTools(df, "photos" if photo_mode else "webcam")
                agent = Agent(GeminiLLM(), tools)
                text, trace = agent.run(
                    "Analyse this session and write a short report.")
                st.session_state.setdefault("ai_reports", {})[report_key] = (
                    text, trace)
            except SystemExit as e:        # for example: API key missing
                st.error(str(e))
            except Exception as e:
                st.error(f"The AI agent could not run: {e}")

    saved = st.session_state.get("ai_reports", {}).get(report_key)
    if saved:
        text, trace = saved
        st.markdown(text)
        with st.expander("🔧 Tool calls the agent made"):
            for name, args, result in trace:
                st.markdown(f"**{name}** `{args}`")
                st.json(result)

with tab6:
    st.subheader("💬 Ask the agent")
    st.caption("Ask a question about this session. The agent answers from "
               "the session data and declines questions the data cannot answer.")

    history = st.session_state.setdefault(f"chat|{report_key}", [])
    for q, a in history:
        st.markdown(f"**You:** {q}")
        st.markdown(a)
        st.divider()

    question = st.text_input(
        "Your question", placeholder="e.g. When did the emotion change?")

    if st.button("Ask") and question.strip():
        with st.spinner("The agent is thinking..."):
            try:
                from agent import Agent
                from llm_gemini import GeminiLLM
                from tools import SessionTools

                agent = Agent(GeminiLLM(), SessionTools(
                    df, "photos" if photo_mode else "webcam"))
                answer, _ = agent.run(question.strip())
                history.append((question.strip(), answer))
                st.rerun()
            except SystemExit as e:        # for example: API key missing
                st.error(str(e))
            except Exception as e:
                st.error(f"The AI agent could not run: {e}")

# ---------------- Recording coach ----------------
st.divider()
with st.expander("🎥 Recording coach: was this a good recording?"):
    from tools import SessionTools

    coach_tools = SessionTools(df, "photos" if photo_mode else "webcam")
    q = coach_tools.get_recording_quality()

    if min_conf > 0:
        st.info("Set 'Minimum confidence' to 0 in the sidebar for an "
                "accurate quality check.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Average confidence", f"{q['average_confidence']:.2f}")
    m2.metric("Below 0.5 confidence",
              f"{q['percent_below_0_5_confidence']:.0f}%")
    if "face_detected_percent" in q:
        m3.metric("Face detected", f"{q['face_detected_percent']:.0f}%")
    else:
        m3.metric("Photos", q["num_predictions"])

    # The checks below come from code, so they work without an API key
    if q["issues"]:
        for issue in q["issues"]:
            st.warning(f"{issue['meaning']}  \n**Tip:** {issue['tip']}")
    else:
        st.success("No recording problems found.")
    st.caption(q["note"])

    # Optional: the agent explains the issues in plain words
    coach_key = (f"coach|{uploaded.name if uploaded is not None else chosen}"
                 f"|{len(df)}|{min_conf}")
    if st.button("Ask the agent for coaching tips", key="coach_button"):
        with st.spinner("The agent is checking the recording..."):
            try:
                from agent import COACH_PROMPT, Agent
                from llm_gemini import GeminiLLM

                coach = Agent(GeminiLLM(), coach_tools,
                              system_prompt=COACH_PROMPT)
                coach_text, _ = coach.run(
                    "How good was this recording? How can I improve it?")
                st.session_state.setdefault("coach_reports", {})[coach_key] = (
                    coach_text)
            except SystemExit as e:          # for example: API key missing
                st.error(str(e))
            except Exception as e:
                st.error(f"The AI agent could not run: {e}")

    coach_saved = st.session_state.get("coach_reports", {}).get(coach_key)
    if coach_saved:
        st.markdown(coach_saved)