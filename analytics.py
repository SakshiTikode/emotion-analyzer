"""Analyze session_log.csv: summary, emotion segments and a timeline plot.

These functions will later become the tools of the AI agent.
"""

import pandas as pd

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_log(path="session_log.csv"):
    """Load the log and add a 'seconds' column (time since session start)."""
    df = pd.read_csv(path)
    if not df.empty:
        df["seconds"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df

def df_from_rows(rows):
    """Build a log DataFrame from (timestamp, emotion, confidence) tuples (e.g. photo results)."""
    df = pd.DataFrame(rows, columns=["timestamp", "emotion", "confidence"])
    if not df.empty:
        df["seconds"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df

def emotion_summary(df, min_conf=0.0):
    """Percentage of predictions per emotion (optionally ignore low-confidence ones)."""
    d = df[df["confidence"] >= min_conf]
    return (d["emotion"].value_counts(normalize=True) * 100).round(1).to_dict()


def emotion_segments(df):
    """Group consecutive identical predictions into segments with start/end/duration."""
    run_id = (df["emotion"] != df["emotion"].shift()).cumsum()
    segs = df.groupby(run_id).agg(
        emotion=("emotion", "first"),
        start=("seconds", "first"),
        mean_conf=("confidence", "mean"),
    )
    segs["end"] = segs["start"].shift(-1)                       # next segment starts here
    segs.loc[segs.index[-1], "end"] = df["seconds"].iloc[-1]    # last segment ends at the last sample
    segs["duration"] = segs["end"] - segs["start"]
    return segs[["emotion", "start", "end", "duration", "mean_conf"]].round(2).reset_index(drop=True)


def plot_timeline(df, path="session_timeline.png"):
    """Plot the predicted emotion over time as a step chart."""
    import matplotlib.pyplot as plt

    y = df["emotion"].map({e: i for i, e in enumerate(EMOTIONS)})
    plt.figure(figsize=(9, 3.5))
    plt.step(df["seconds"], y, where="post")
    plt.yticks(range(len(EMOTIONS)), EMOTIONS)
    plt.xlabel("Seconds since start")
    plt.title("Emotion timeline")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def clean_segments(segs, min_duration=0.5):
    """Remove very short segments (likely noise) and merge neighbours that become identical.

    Example: neutral -> sad (0.2 s) -> neutral becomes one long neutral segment.
    """
    kept = segs[segs["duration"] >= min_duration].reset_index(drop=True)
    if kept.empty:
        return kept

    merged = []
    for _, row in kept.iterrows():
        if merged and merged[-1]["emotion"] == row["emotion"]:
            prev = merged[-1]
            total = prev["duration"] + row["duration"]
            # duration-weighted average of the confidences
            prev["mean_conf"] = (prev["mean_conf"] * prev["duration"]
                                 + row["mean_conf"] * row["duration"]) / total
            prev["end"] = row["end"]
            prev["duration"] = prev["end"] - prev["start"]
        else:
            merged.append(row.to_dict())
    return pd.DataFrame(merged).round(2)


if __name__ == "__main__":
    df = load_log()
    if df.empty:
        raise SystemExit("session_log.csv is empty. Run webcam_app.py first.")

    print(f"Predictions: {len(df)} | Duration: {df['seconds'].iloc[-1]:.1f} s")
    print("\nEmotion summary (% of predictions):")
    print(emotion_summary(df))

    segs = emotion_segments(df)
    cleaned = clean_segments(segs, min_duration=0.5)
    print(f"\nRaw segments: {len(segs)} | After cleaning: {len(cleaned)}")
    print("\nCleaned segments:")
    print(cleaned.to_string())
    plot_timeline(df)