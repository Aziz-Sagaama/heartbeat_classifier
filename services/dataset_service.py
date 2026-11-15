from __future__ import annotations

from typing import Any


DATASET_STATS: dict[str, Any] = {
    "name": "MIT-BIH Arrhythmia Database",
    "sampling_frequency_hz": 360,
    "recordings": 48,
    "heartbeat_samples": 234,
    "input_shape": [234, 1],
    "total_heartbeats": 101413,
    "original_classes": {
        "N": 90595,
        "S": 2781,
        "V": 7235,
        "F": 802,
    },
    "binary_classes": {"N": 90595, "AN": 10818},
    "split": {
        "DS1": "Patient-level development and cross-validation split.",
        "DS2": (
            "Patient-level evaluation split used during model comparison and "
            "development; it is not a completely untouched test set."
        ),
    },
}


def get_dataset_stats() -> dict[str, Any]:
    return DATASET_STATS
