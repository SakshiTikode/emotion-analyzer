"""Data loading, class weights and augmentation for the FER2013 emotion project."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (48, 48)   # FER2013 images are 48x48 pixels
BATCH = 64
SEED = 42             # same seed for train and validation so they never overlap


def load_datasets():
    """Return (train_ds, val_ds, test_ds) as batched tf.data datasets."""
    common = dict(
        label_mode="categorical",   # one-hot labels, e.g. happy = [0,0,0,1,0,0,0]
        color_mode="grayscale",     # 1 channel
        image_size=IMG_SIZE,
        batch_size=BATCH,
        seed=SEED,
    )
    # 85% of data/train is used for training ...
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "data/train", validation_split=0.15, subset="training", **common)
    # ... and the other 15% for validation (same seed = no overlap)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "data/train", validation_split=0.15, subset="validation", **common)
    # The test set is never used for training or model selection.
    # shuffle=False keeps predictions in the same order as the labels.
    test_ds = tf.keras.utils.image_dataset_from_directory(
        "data/test", shuffle=False, **common)
    return train_ds, val_ds, test_ds


def get_class_weights(train_ds, power=0.5):
    """Compute class weights to reduce the effect of class imbalance.

    Rare classes (e.g. disgust) get a larger weight, common ones (e.g. happy) a
    smaller one. power=0.5 takes the square root of the balanced weights.
    Full weights (power=1.0) were too aggressive and stopped training.
    """
    labels = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in train_ds])
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(c): float(w) ** power for c, w in zip(classes, weights)}


def get_augmentation():
    """Random flips/rotations/zoom. Active only during training."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])