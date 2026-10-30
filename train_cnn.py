"""Train the baseline CNN on FER2013.

Note: this was trained on a GPU (Colab). On a laptop CPU one epoch takes
about 4-5 minutes, so 40 epochs is slow.
"""

import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

from data_loader import get_augmentation, get_class_weights, load_datasets

EPOCHS = 40
LEARNING_RATE = 3e-4   # 1e-3 was too high together with strong class weights
os.makedirs("model", exist_ok=True)

# ---------------- Data ----------------
train_ds, val_ds, _ = load_datasets()
class_weights = get_class_weights(train_ds)   # square-root (soft) weights
print("Class weights:", class_weights)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)   # load the next batch while training
val_ds = val_ds.prefetch(AUTOTUNE)


# ---------------- Model ----------------
def conv_block(x, filters):
    """Two conv layers + batch norm, then downsample and apply dropout."""
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)      # halves width and height
    x = layers.Dropout(0.25)(x)       # randomly drops neurons to reduce overfitting
    return x


inputs = layers.Input(shape=(48, 48, 1))
x = get_augmentation()(inputs)         # only active during training
x = layers.Rescaling(1.0 / 255)(x)     # pixel values 0-255 -> 0-1
x = conv_block(x, 32)                  # learns edges and simple shapes
x = conv_block(x, 64)                  # learns parts such as eyes and mouth
x = conv_block(x, 128)                 # learns higher-level facial patterns
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(7, activation="softmax")(x)   # 7 emotion probabilities

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---------------- Callbacks ----------------
cbs = [
    # Keep the model with the best validation accuracy
    callbacks.ModelCheckpoint("model/best_model.keras",
                              monitor="val_accuracy", save_best_only=True),
    # Stop if validation loss has not improved for 8 epochs
    callbacks.EarlyStopping(monitor="val_loss", patience=8,
                            restore_best_weights=True),
    # Halve the learning rate if validation loss stalls for 3 epochs
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
]

# ---------------- Train ----------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=cbs,
)

with open("model/history.json", "w") as f:
    json.dump(history.history, f)

# ---------------- Training curves ----------------
h = history.history
plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(h["loss"], label="train")
plt.plot(h["val_loss"], label="val")
plt.title("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(h["accuracy"], label="train")
plt.plot(h["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("model/training_curves.png")
plt.show()