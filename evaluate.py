"""Evaluate a trained model on the test set.

Usage:
    python evaluate.py                  # baseline CNN (model/best_model.keras)
    python evaluate.py best_mobilenet   # transfer learning model
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# The model name comes from the command line (default: baseline)
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "best_model"
MODEL_PATH = f"model/{MODEL_NAME}.keras"
print("Model:", MODEL_PATH)

# shuffle=False so predictions line up with the true labels
test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=(48, 48),
    batch_size=64,
    shuffle=False,
)

model = tf.keras.models.load_model(MODEL_PATH)

# Predict on the whole test set
y_true, y_pred = [], []
for x, y in test_ds:
    probs = model.predict(x, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1))
    y_true.extend(np.argmax(y.numpy(), axis=1))
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Overall accuracy and per-class precision / recall / F1
acc = (y_true == y_pred).mean()
print(f"\nTest accuracy: {acc:.4f}\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                            digits=3, zero_division=0))

# Confusion matrix: counts, and row-normalized (each row sums to 1)
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax[0])
ax[0].set_title(f"Confusion matrix (counts) - {MODEL_NAME}")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax[1])
ax[1].set_title("Confusion matrix (row-normalized)")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"model/confusion_matrix_{MODEL_NAME}.png")
plt.show()