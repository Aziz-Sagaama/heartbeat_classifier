
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

from model import build_model


# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load prepared data
data = np.load("src/data/prepared_data.npz")

X_train = data["X_train"]
y_train = data["y_train"]
train_groups = data["train_groups"]
classes = data["classes"]

# Create a recording-independent validation split
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fit_idx, val_idx = next(
    splitter.split(X_train, y_train, train_groups)
)

X_fit = X_train[fit_idx]
y_fit = y_train[fit_idx]

X_val = X_train[val_idx]
y_val = y_train[val_idx]

print("Training examples:", len(X_fit))
print("Validation examples:", len(X_val))

print("Training class distribution:", Counter(y_fit))
print("Validation class distribution:", Counter(y_val))

# Calculate class weights using training examples only
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_fit),
    y=y_fit
)

class_weight_dict = {
    int(class_id): float(weight)
    for class_id, weight in zip(np.unique(y_fit), class_weights)
}

print("\nClass mapping:", dict(enumerate(classes)))
print("Class weights:", class_weight_dict)

# Build CNN
model = build_model()

# Create model directory
os.makedirs("models", exist_ok=True)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "models/heartbeat_cnn.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# Train
history = model.fit(
    X_fit,
    y_fit,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=256,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining completed!")
print("Best model saved to models/heartbeat_cnn.keras")
