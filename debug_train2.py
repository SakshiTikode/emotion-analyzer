import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from data_loader import load_datasets

train_ds, val_ds, _ = load_datasets()
class_names = train_ds.class_names
train_small = train_ds.take(100).prefetch(tf.data.AUTOTUNE)
val_small = val_ds.take(30).prefetch(tf.data.AUTOTUNE)

with open("class_weights.json") as f:
    full_w = {int(k): v for k, v in json.load(f).items()}
soft_w = {k: v ** 0.5 for k, v in full_w.items()}
print("Soft weights:", {class_names[k]: round(v, 2) for k, v in soft_w.items()})

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    return x

def build_model(lr):
    inputs = layers.Input(shape=(48, 48, 1))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(7, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def predicted_distribution(model, ds):
    preds = []
    for x, _ in ds:
        preds.extend(np.argmax(model.predict(x, verbose=0), axis=1))
    counts = np.bincount(preds, minlength=7)
    print("Model ne predict kela:", dict(zip(class_names, counts.tolist())))

experiments = [
    ("D: no weights, lr=1e-3", 1e-3, None),
    ("E: soft weights, lr=3e-4", 3e-4, soft_w),
]

for name, lr, cw in experiments:
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    model = build_model(lr)
    model.fit(train_small, validation_data=val_small,
              epochs=5, class_weight=cw, verbose=2)
    predicted_distribution(model, val_small)