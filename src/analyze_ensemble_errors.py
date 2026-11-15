
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_PATH = "src/data/processed/ensemble_ds2_results.npz"
THRESHOLD = 0.87

data = np.load(DATA_PATH)

y_true = data["y_true"].astype(int)
y_prob = data["y_prob"].astype(float)
groups = data["groups"].astype(str)

y_pred = (y_prob >= THRESHOLD).astype(int)

results = []

for patient in np.unique(groups):
    mask = groups == patient

    yt = y_true[mask]
    yp = y_pred[mask]

    tp = np.sum((yt == 1) & (yp == 1))
    fp = np.sum((yt == 0) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))
    tn = np.sum((yt == 0) & (yp == 0))

    results.append({
        "Patient": patient,
        "Total": len(yt),
        "Abnormal": int(np.sum(yt == 1)),
        "Abnormal_F1": round(f1_score(yt, yp, zero_division=0), 4),
        "Precision": round(precision_score(yt, yp, zero_division=0), 4),
        "Recall": round(recall_score(yt, yp, zero_division=0), 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    })

df = pd.DataFrame(results)

df = df.sort_values("Abnormal_F1")

print("\n========== PATIENT-LEVEL RESULTS ==========")
print(df.to_string(index=False))

print("\n========== PATIENTS WITH MOST FALSE NEGATIVES ==========")
print(
    df.sort_values("FN", ascending=False)[
        ["Patient", "Abnormal", "FN", "Recall", "Abnormal_F1"]
    ].head(10).to_string(index=False)
)

print("\n========== PATIENTS WITH MOST FALSE POSITIVES ==========")
print(
    df.sort_values("FP", ascending=False)[
        ["Patient", "FP", "Precision", "Abnormal_F1"]
    ].head(10).to_string(index=False)
)

df.to_csv("src/data/processed/ensemble_patient_analysis.csv", index=False)

print("\nSaved: src/data/processed/ensemble_patient_analysis.csv")
