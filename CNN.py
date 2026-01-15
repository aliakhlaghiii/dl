# Cats vs Dogs (CNN)

import tensorflow as tf
from tensorflow import keras as krs
from matplotlib import pyplot as plt

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

IMG_SIZE = (112, 112)
BATCH_SIZE = 20

# Download + unzip dataset into a local folder structure (train/ and test/)
%%bash
set -e
curl -L https://bioinf.nl/~davelangers/datasets/dogs-vs-cats.zip -o dogs-vs-cats.zip
unzip -oq dogs-vs-cats.zip
ls -la dogs-vs-cats

# Create train/val from the same train folder using a fixed split + seed; create test separately (no shuffle)
train_ds = krs.utils.image_dataset_from_directory(
    "dogs-vs-cats/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.25,
    subset="training",
)

val_ds = krs.utils.image_dataset_from_directory(
    "dogs-vs-cats/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.25,
    subset="validation",
)

test_ds = krs.utils.image_dataset_from_directory(
    "dogs-vs-cats/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Show the folder-name-to-label mapping (DON'T assume 0=cat/1=dog without checking this)
print("Class names:", train_ds.class_names)

# Speed up input pipeline: keep data in memory/disk cache and overlap preprocessing with GPU/CPU via prefetch
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# Take one batch to inspect tensor shapes/dtypes (x: images, y: labels)
for x, y in train_ds.take(1):
    break

print("x.shape =", x.shape)
print("y.shape =", y.shape)
print("x.dtype =", x.dtype)
print("y.dtype =", y.dtype)

# Display one image; if pixel values are 0..255 in float, rescale to 0..1 to avoid imshow clipping warnings
img = x[0].numpy()
img_show = img / 255.0 if img.max() > 1.5 else img

plt.imshow(img_show)
plt.title(f"y = {int(y[0])}")
plt.axis("off")
plt.show()

# Build a preprocessing block: normalize (1/255) + random augmentations (augmentations only when training=True)
preproc = krs.Sequential([
    krs.layers.InputLayer(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input"),
    krs.layers.Rescaling(1./255),
    krs.layers.RandomFlip("horizontal"),
    krs.layers.RandomContrast(0.5),
    krs.layers.RandomRotation(0.05),
    krs.layers.RandomZoom(height_factor=(-0.1, 0.1)),
], name="preproc")

# Visualize augmentation by repeatedly applying preproc in training mode (outputs vary each call)
for x, y in train_ds.take(1):
    break

plt.figure(figsize=(10, 8))
for i in range(12):
    x_aug = preproc(x, training=True)
    plt.subplot(3, 4, i+1)
    plt.imshow(x_aug[2].numpy())
    plt.title(f"y={int(y[2])}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# Define the CNN: conv/pool feature extractor -> flatten -> dense classifier -> sigmoid for binary output
model = krs.Sequential([
    preproc,

    krs.layers.Conv2D(32, (3, 3), activation="relu"),
    krs.layers.Conv2D(32, (3, 3), activation="relu"),
    krs.layers.SpatialDropout2D(0.1),
    krs.layers.MaxPooling2D(pool_size=(2, 2)),

    krs.layers.Conv2D(64, (3, 3), activation="relu"),
    krs.layers.Conv2D(64, (5, 5), activation="relu"),
    krs.layers.MaxPooling2D(pool_size=(3, 3)),

    krs.layers.Flatten(),
    krs.layers.Dropout(0.1),

    krs.layers.Dense(
        64,
        activation="tanh",
        kernel_regularizer=krs.regularizers.L1L2(l1=1e-5, l2=1e-4),
    ),
    krs.layers.Dropout(0.1),

    krs.layers.Dense(16, activation="tanh"),
    krs.layers.Dense(1, activation="sigmoid"),
], name="dogs_vs_cats_cnn")

# Print architecture + parameter counts
model.summary()

# Configure training for binary classification with sigmoid output
model.compile(
    loss=krs.losses.BinaryCrossentropy(),
    optimizer=krs.optimizers.Adam(),
    metrics=[krs.metrics.BinaryAccuracy()],
)

print(model.metrics_names)

# Callbacks: stop early if val_loss stops improving; reduce LR on plateau; save best model by val_loss
callbacks = [
    krs.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    krs.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    ),
    krs.callbacks.ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
]

# Train on train_ds and validate on val_ds (test_ds is only for the final report)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks,
)

# Final evaluation on test set (one-time, unbiased)
test_loss, test_acc = model.evaluate(test_ds)
print("Test loss:", test_loss)
print("Test acc :", test_acc)

# Plot learning curves to see overfit/underfit trends across epochs
plt.figure()
plt.plot(history.history.get("loss", []), label="train_loss")
plt.plot(history.history.get("val_loss", []), label="val_loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

plt.figure()
plt.plot(history.history.get("binary_accuracy", []), label="train_acc")
plt.plot(history.history.get("val_binary_accuracy", []), label="val_acc")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()
