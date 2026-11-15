
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

data = np.load("src/data/processed/binary_oof_stratified.npz")

y_true = data["y_true"]
y_prob = data["y_prob"]

# Handle either (n,) or (n, 2) probability arrays
if y_prob.ndim == 2:
    abnormal_prob = y_prob[:, 1]
else:
    abnormal_prob = y_prob

results = []

for threshold in np.arange(0.10, 0.91, 0.01):
    y_pred = (abnormal_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results.append((f1, threshold, precision, recall))

results.sort(reverse=True)

print("\nTOP 10 THRESHOLDS — ABNORMAL CLASS")
print("-" * 65)
print(f"{'Threshold':<12}{'F1':<12}{'Precision':<12}{'Recall':<12}")

for f1, threshold, precision, recall in results[:10]:
    print(f"{threshold:<12.2f}{f1:<12.4f}{precision:<12.4f}{recall:<12.4f}")

print("\nBest threshold:", round(results[0][1], 2))
