"""Shared constants and cached loaders used by all pages."""

import streamlit as st
import hashlib
import time

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# One colour and one emoji per emotion, used consistently everywhere
COLORS = {
    "angry": "#E74C3C", "disgust": "#8BC34A", "fear": "#9B59B6",
    "happy": "#F1C40F", "neutral": "#95A5A6", "sad": "#3498DB",
    "surprise": "#FF9800",
}
EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😊",
    "neutral": "😐", "sad": "😢", "surprise": "😮",
}

MODEL_PATH = "model/best_mobilenet.keras"


@st.cache_resource   # load the model only once, not on every interaction
def load_model():
    import tensorflow as tf   # imported here so pages without a model load fast
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_detector():
    import cv2
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def log_photo_result(photo_bytes, emotion, confidence):
    """Add one photo result to this browser session (only once per photo)."""
    photo_id = hashlib.md5(photo_bytes).hexdigest()
    if st.session_state.get("last_photo_id") == photo_id:
        return False   # Streamlit reran with the same photo, do not add it twice
    st.session_state["last_photo_id"] = photo_id
    st.session_state.setdefault("log", []).append(
        (time.time(), emotion, float(confidence)))
    return True