import numpy as np
import tensorflow as tf
import keras

print("TF:", tf.__version__, "| Keras:", keras.__version__)

common = dict(label_mode="categorical", color_mode="grayscale",
              image_size=(48, 48), batch_size=64, seed=42)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train", validation_split=0.15, subset="validation", **common)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test", shuffle=False, **common)

model = tf.keras.models.load_model("model/best_model.keras")

loss, acc = model.evaluate(val_ds, verbose=0)
print(f"VAL  -> loss {loss:.3f}, accuracy {acc:.3f}")
loss, acc = model.evaluate(test_ds, verbose=0)
print(f"TEST -> loss {loss:.3f}, accuracy {acc:.3f}")

x, _ = next(iter(test_ds))
print("Input dtype/min/max:", x.dtype, float(tf.reduce_min(x)), float(tf.reduce_max(x)))
print("Sample predictions (3 images):")
print(model.predict(x[:3], verbose=0).round(2))