import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
)

PATH = "src/data/processed/oof_predictions.npz"
NAMES = ["N", "S", "V", "F"]

data = np.load(PATH)

y_true = data["y_true"]
y_pred = data["y_pred"]
y_prob = data["y_prob"]

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_true,
    y_pred,
    target_names=NAMES,
    digits=4,
    zero_division=0
))

print("\n========== MEAN PROBABILITIES BY TRUE CLASS ==========")
for class_id, name in enumerate(NAMES):
    mask = y_true == class_id
    means = y_prob[mask].mean(axis=0)

    print(f"\nActual class: {name} ({mask.sum()} beats)")
    for predicted_id, predicted_name in enumerate(NAMES):
        print(f"  P({predicted_name}): {means[predicted_id]:.4f}")

print("\n========== CLASS PROBABILITY DIAGNOSTICS ==========")
for class_id, name in enumerate(NAMES):
    binary_true = (y_true == class_id).astype(int)
    scores = y_prob[:, class_id]

    precision, recall, thresholds = precision_recall_curve(
        binary_true, scores
    )

    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-10
    )

    best = np.argmax(f1)

    print(f"\n{name}:")
    print(f"  Average Precision: {average_precision_score(binary_true, scores):.4f}")
    print(f"  Best diagnostic threshold: {thresholds[best]:.4f}")
    print(f"  Diagnostic precision: {precision[best]:.4f}")
    print(f"  Diagnostic recall: {recall[best]:.4f}")
    print(f"  Diagnostic F1: {f1[best]:.4f}")

print("\n========== PREDICTION DISTRIBUTION ==========")
for class_id, name in enumerate(NAMES):
    print(f"{name}: {(y_pred == class_id).sum()}")