from __future__ import annotations

from typing import Any


ENSEMBLE_EVALUATION: dict[str, Any] = {
    "name": "Five-model CNN ensemble on DS2",
    "threshold": 0.87,
    "metrics": {
        "accuracy": 0.92,
        "macro_f1": 0.7526,
        "abnormal_f1": 0.5510,
        "abnormal_precision": 0.67,
        "abnormal_recall": 0.47,
        "normal_specificity": 0.97,
    },
    "per_class": {
        "N": {"precision": 0.94, "recall": 0.97, "f1": 0.95},
        "AN": {"precision": 0.67, "recall": 0.47, "f1": 0.55},
    },
    "confusion_matrix": {
        "labels": ["N", "AN"],
        "rows": [
            {"actual": "N", "predicted_N": 43018, "predicted_AN": 1222},
            {"actual": "AN", "predicted_N": 2910, "predicted_AN": 2535},
        ],
    },
    "methodology": (
        "The ensemble averages abnormal probabilities from five CNN models "
        "trained on DS1 cross-validation folds."
    ),
}

DS1_CV_EVALUATION: dict[str, Any] = {
    "name": "Original binary CNN DS1 out-of-fold evaluation",
    "macro_f1": 0.7171,
    "abnormal_f1": 0.5146,
    "abnormal_recall": 0.7256,
    "normal_specificity": 0.8772,
}


def get_evaluation() -> dict[str, Any]:
    return {
        "ensemble": ENSEMBLE_EVALUATION,
        "ds1_cross_validation": DS1_CV_EVALUATION,
    }
