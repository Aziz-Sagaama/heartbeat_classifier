
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "final_splits.npz"
MODEL_PATH = BASE_DIR.parent / "models" / "best_model.keras"

CLASS_NAMES = ["N", "S", "V", "F"]

# Load test data
data = np.load(DATA_PATH)

X_test = data["X_test"].astype(np.float32)

CLASS_TO_INDEX = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3
}

y_test = np.array([
    CLASS_TO_INDEX[label] for label in data["y_test"]
])

print("Test data shape:", X_test.shape)

# Load the best saved model
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=1
)

# Predictions
probabilities = model.predict(X_test, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

print("\n========== TEST RESULTS ==========")
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test loss: {test_loss:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3]
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Heartbeat Classification - Confusion Matrix")
plt.tight_layout()

OUTPUT_PATH = BASE_DIR.parent / "models" / "confusion_matrix.png"
plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()

print("\nConfusion matrix saved at:", OUTPUT_PATH)
