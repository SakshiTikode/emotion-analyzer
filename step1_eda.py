import os
import random
import cv2
import matplotlib.pyplot as plt

DATA_DIR = "data"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def count_images(split):
    counts = {}
    for emo in EMOTIONS:
        folder = os.path.join(DATA_DIR, split, emo)
        counts[emo] = len(os.listdir(folder)) if os.path.exists(folder) else 0
    return counts

train_counts = count_images("train")
test_counts = count_images("test")
print("Train:", train_counts)
print("Test:", test_counts)

# Class distribution graph
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(train_counts.keys(), train_counts.values(), color="steelblue")
ax[0].set_title("Train distribution")
ax[1].bar(test_counts.keys(), test_counts.values(), color="coral")
ax[1].set_title("Test distribution")
for a in ax:
    a.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

# Sample images (3 per emotion)
fig, axes = plt.subplots(len(EMOTIONS), 3, figsize=(6, 14))
for i, emo in enumerate(EMOTIONS):
    folder = os.path.join(DATA_DIR, "train", emo)
    files = random.sample(os.listdir(folder), 3)
    for j, f in enumerate(files):
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        axes[i, j].imshow(img, cmap="gray")
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_title(emo, fontsize=10, loc="left")
plt.tight_layout()
plt.savefig("samples.png")
plt.show()

# Image shape
first = os.listdir(os.path.join(DATA_DIR, "train", "happy"))[0]
img = cv2.imread(os.path.join(DATA_DIR, "train", "happy", first), cv2.IMREAD_GRAYSCALE)
print("Image shape:", img.shape)