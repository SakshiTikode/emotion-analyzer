import json
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import load_datasets, get_class_weights, get_augmentation

train_ds, val_ds, test_ds = load_datasets()
class_names = train_ds.class_names
print("Classes:", class_names)

# Class weights
weights = get_class_weights(train_ds)
print("Class weights:")
for i, name in enumerate(class_names):
    print(f"  {name}: {weights[i]:.2f}")
with open("class_weights.json", "w") as f:
    json.dump(weights, f)

# Augmentation demo: ek image cha 6 versions
augment = get_augmentation()
images, labels = next(iter(train_ds))
img = images[0:1]

plt.figure(figsize=(10, 3))
plt.subplot(1, 6, 1)
plt.imshow(img[0, :, :, 0], cmap="gray")
plt.title("original")
plt.axis("off")
for i in range(5):
    aug = augment(img, training=True)
    plt.subplot(1, 6, i + 2)
    plt.imshow(aug[0, :, :, 0], cmap="gray")
    plt.title(f"aug {i+1}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("augmentation_demo.png")
plt.show()