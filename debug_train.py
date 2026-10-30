import json
import tensorflow as tf
from tensorflow.keras import layers, models
from data_loader import load_datasets, get_augmentation

train_ds, val_ds, _ = load_datasets()
train_small = train_ds.take(100).prefetch(tf.data.AUTOTUNE)
val_small = val_ds.take(30).prefetch(tf.data.AUTOTUNE)

with open("class_weights.json") as f:
    class_weights = {int(k): v for k, v in json.load(f).items()}

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    return x

def build_model(use_aug):
    inputs = layers.Input(shape=(48, 48, 1))
    x = get_augmentation()(inputs) if use_aug else inputs
    x = layers.Rescaling(1.0 / 255)(x)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(7, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

experiments = [
    ("A: plain (no aug, no class weights)", False, None),
    ("B: + class weights", False, class_weights),
    ("C: + augmentation", True, class_weights),
]

for name, use_aug, cw in experiments:
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    model = build_model(use_aug)
    model.fit(train_small, validation_data=val_small,
              epochs=3, class_weight=cw, verbose=2)