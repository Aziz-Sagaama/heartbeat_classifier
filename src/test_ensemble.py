
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import classification_report, f1_score, confusion_matrix

DATA_PATH = "src/data/processed/mitbih_dataset.npz"
MODEL_DIR = Path("models/binary_stratified")

DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234"
]

# Load DS2
data = np.load(DATA_PATH)

X = data["X"].astype(np.float32)
y_original = data["y"]
groups = data["groups"].astype(str)

mask = np.isin(groups, DS2)

X_test = X[mask]
y_test = (y_original[mask] != "N").astype(np.int32)

# Load and predict with the five CV models
predictions = []

for fold in range(1, 6):
    model_path = MODEL_DIR / f"fold_{fold}.keras"

    print(f"Loading {model_path}...")
    model = tf.keras.models.load_model(model_path)

    probabilities = model.predict(
        X_test, batch_size=256, verbose=0
    )

    predictions.append(probabilities[:, 1])

    del model
    tf.keras.backend.clear_session()

# Average the five models' abnormal probabilities
ensemble_prob = np.mean(predictions, axis=0)

print("\nENSEMBLE RESULTS")

for threshold in [0.50, 0.87]:
    y_pred = (ensemble_prob >= threshold).astype(np.int32)

    print(f"\n========== THRESHOLD {threshold} ==========")

    print(classification_report(
        y_test,
        y_pred,
        target_names=["N", "AN"],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Abnormal F1:",
          f1_score(y_test, y_pred, zero_division=0))

    print("Macro F1:",
          f1_score(y_test, y_pred, average="macro"))

np.savez(
    "src/data/processed/ensemble_ds2_results.npz",
    y_true=y_test,
    y_prob=ensemble_prob,
    groups=groups[mask]
)

print("\nEnsemble predictions saved.")
