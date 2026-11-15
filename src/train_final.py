
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras import layers, models

tf.keras.utils.set_random_seed(42)

DATA_PATH = "src/data/processed/mitbih_dataset.npz"
MODEL_PATH = "models/binary_final.keras"
THRESHOLD = 0.87

DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230"
]

DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234"
]

# Load original dataset
data = np.load(DATA_PATH)

X = data["X"].astype(np.float32)
y_original = data["y"]
groups = data["groups"].astype(str)

# Binary labels: N=0, S/V/F=1
y = (y_original != "N").astype(np.int32)

train_mask = np.isin(groups, DS1)
test_mask = np.isin(groups, DS2)

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print("DS1:", X_train.shape)
print("DS2:", X_test.shape)
print("Training distribution:", np.bincount(y_train))
print("Testing distribution:", np.bincount(y_test))

# Normalize using DS1 only
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(X_train)

# CNN architecture
model = models.Sequential([
    layers.Input(shape=(234, 1)),
    normalizer,

    layers.Conv1D(32, 7, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),

    layers.Conv1D(64, 5, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),

    layers.Conv1D(128, 3, activation="relu"),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling1D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Same class-weight strategy as cross-validation
counts = np.bincount(y_train, minlength=2)
weights = np.sqrt(counts.max() / counts)
weights = np.minimum(weights, 6.0)

class_weights = {
    0: float(weights[0]),
    1: float(weights[1])
}

print("Class weights:", class_weights)

# Train on all DS1 — do not use DS2 for validation
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=256,
    class_weight=class_weights,
    shuffle=True,
    verbose=1
)

Path("models").mkdir(exist_ok=True)
model.save(MODEL_PATH)

# Final independent evaluation on DS2
probabilities = model.predict(X_test, batch_size=256, verbose=1)
abnormal_prob = probabilities[:, 1]

y_pred = (abnormal_prob >= THRESHOLD).astype(np.int32)

print("\nFINAL DS2 RESULTS — THRESHOLD 0.87")
print(classification_report(
    y_test,
    y_pred,
    target_names=["N", "AN"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("Abnormal F1:", f1_score(y_test, y_pred, zero_division=0))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))

np.savez(
    "src/data/processed/final_ds2_results.npz",
    y_true=y_test,
    y_pred=y_pred,
    y_prob=probabilities,
    groups=groups[test_mask]
)

print("\nModel saved:", MODEL_PATH)
print("Results saved: src/data/processed/final_ds2_results.npz")
