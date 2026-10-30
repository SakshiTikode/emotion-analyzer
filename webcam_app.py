import csv
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import datetime
import os

# ---------------- Settings ----------------
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MODEL_PATH = "model/best_mobilenet.keras"
SMOOTH_N = 5         # average the last 5 predictions to reduce flicker
PREDICT_EVERY = 2    # run the model on every 2nd frame to keep it fast

# ---------------- Load model and face detector ----------------
model = tf.keras.models.load_model(MODEL_PATH)

# Haar Cascade: a classic, lightweight face detector that ships with OpenCV 4.x
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise SystemExit("Could not load the face detector.")

# ---------------- Open the webcam ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam. Check camera permission "
                     "and close other apps that use the camera.")

history = deque(maxlen=SMOOTH_N)   # recent probability vectors
log = []                           # (timestamp, emotion, confidence)
frame_count = 0
label_text = ""

print("Running. Press 'q' in the video window to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read a frame from the webcam.")
        break

    # 1. Convert the frame to grayscale (the model was trained on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    if len(faces) > 0:
        # 3. If there are several faces, use the largest one
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        if frame_count % PREDICT_EVERY == 0:
            # 4. Crop the face and resize it to 48x48, the size the model expects
            face = cv2.resize(gray[y:y + h, x:x + w], (48, 48),
                              interpolation=cv2.INTER_AREA)
            inp = face.astype("float32").reshape(1, 48, 48, 1)

            # 5. Predict the emotion probabilities
            probs = model(inp, training=False).numpy()[0]

            # 6. Smooth over the last few predictions to avoid flickering labels
            history.append(probs)
            avg = np.mean(history, axis=0)
            idx = int(np.argmax(avg))
            label_text = f"{CLASS_NAMES[idx]} {avg[idx] * 100:.0f}%"

            # 7. Save the result for the session log
            log.append((time.time(), CLASS_NAMES[idx], float(avg[idx])))

        # 8. Draw the face box and the emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        # No face found: clear the smoothing history and show a message
        history.clear()
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    frame_count += 1
    cv2.imshow("Emotion Analyzer (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- Clean up ----------------
cap.release()
cv2.destroyAllWindows()

# Save this session in its own file, e.g. sessions/session_20261028_173900.csv
if log:
    os.makedirs("sessions", exist_ok=True)
    name = datetime.datetime.now().strftime("sessions/session_%Y%m%d_%H%M%S.csv")
    with open(name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "emotion", "confidence"])
        writer.writerows(log)
    print(f"Saved {len(log)} predictions to {name}")
else:
    print("No predictions were recorded, nothing saved.")