
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

data = np.load("src/data/processed/final_ds2_results.npz")

y_true = data["y_true"]
y_prob = data["y_prob"][:, 1]

for threshold in [0.50, 0.87]:
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n========== THRESHOLD {threshold:.2f} ==========")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["N", "AN"],
        zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Abnormal F1:", f1_score(y_true, y_pred, zero_division=0))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
