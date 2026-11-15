from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
THRESHOLD = 0.80
MODEL_COUNT = 5


class ModelService:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.models: list[Any] = []
        self.load_error: str | None = None

    @property
    def available(self) -> bool:
        return len(self.models) == MODEL_COUNT

    def load(self) -> None:
        paths = [self.model_dir / f"fold_{fold}.keras" for fold in range(1, MODEL_COUNT + 1)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            self.load_error = (
                "Trained ensemble models are unavailable. Place fold_1.keras through "
                f"fold_{MODEL_COUNT}.keras in {self.model_dir}."
            )
            LOGGER.warning("%s Missing: %s", self.load_error, ", ".join(missing))
            return

        try:
            import tensorflow as tf

            self.models = [tf.keras.models.load_model(path, compile=False) for path in paths]
            self.load_error = None
            LOGGER.info("Loaded %d ensemble models from %s", len(self.models), self.model_dir)
        except (ImportError, OSError, ValueError) as exc:
            self.load_error = f"Unable to load the trained ensemble models: {exc}"
            LOGGER.exception(self.load_error)

    def predict(self, signal: np.ndarray) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError(self.load_error or "The trained ensemble is unavailable.")

        batch = signal.astype(np.float32, copy=False).reshape(1, 234, 1)
        individual = [float(model.predict(batch, verbose=0)[0][1]) for model in self.models]
        ensemble_probability = float(np.mean(individual))
        predicted_class = "AN" if ensemble_probability >= THRESHOLD else "N"
        return {
            "predicted_class": predicted_class,
            "label": "Abnormal" if predicted_class == "AN" else "Normal",
            "abnormal_probability": ensemble_probability,
            "normal_probability": 1.0 - ensemble_probability,
            "threshold": THRESHOLD,
            "individual_model_probabilities": individual,
        }

    def info(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "model_count": MODEL_COUNT,
            "model_directory": str(self.model_dir),
            "input_shape": [234, 1],
            "threshold": THRESHOLD,
            "architecture": [
                {"name": "Input", "details": "234 × 1 ECG signal"},
                {"name": "Normalization", "details": "Training-adapted Keras normalization"},
                {"name": "Conv1D", "details": "32 filters · kernel 7 · ReLU"},
                {"name": "Batch Normalization", "details": ""},
                {"name": "MaxPooling1D", "details": "Pool size 2"},
                {"name": "Conv1D", "details": "64 filters · kernel 5 · ReLU"},
                {"name": "Batch Normalization", "details": ""},
                {"name": "MaxPooling1D", "details": "Pool size 2"},
                {"name": "Conv1D", "details": "128 filters · kernel 3 · ReLU"},
                {"name": "Batch Normalization", "details": ""},
                {"name": "Global Average Pooling", "details": ""},
                {"name": "Dense", "details": "128 neurons · ReLU"},
                {"name": "Dropout", "details": "0.4"},
                {"name": "Output", "details": "2 neurons · Softmax"},
            ],
            "load_error": self.load_error,
        }
