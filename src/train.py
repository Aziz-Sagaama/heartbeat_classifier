
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from model import build_model


DATA_PATH = Path("src/data/processed/final_splits.npz")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "heartbeat_classifier.keras"

BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


class BalancedECGSequence(tf.keras.utils.Sequence):

    def __init__(self, X, y, batch_size=64, **kwargs):
        super().__init__(**kwargs)

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int32)
        self.batch_size = batch_size

        self.classes = np.unique(y)
        self.samples_per_class = batch_size // len(self.classes)

        if batch_size % len(self.classes) != 0:
            raise ValueError("Batch size must be divisible by 4.")

        self.class_indices = {
            cls: np.where(y == cls)[0]
            for cls in self.classes
        }

        self.steps = int(np.ceil(
            max(len(indices) for indices in self.class_indices.values())
            / self.samples_per_class
        ))

        self.rng = np.random.default_rng(RANDOM_STATE)

    def __len__(self):
        return self.steps

    def augment(self, signal):
        signal = signal.copy()

        # Small amplitude variation
        signal *= self.rng.uniform(0.90, 1.10)

        # Small temporal shift with zero padding (no circular wrapping)
        shift = self.rng.integers(-5, 6)
        if shift > 0:
            signal[shift:] = signal[:-shift].copy()
            signal[:shift] = 0
        elif shift < 0:
            shift = abs(shift)
            signal[:-shift] = signal[shift:].copy()
            signal[-shift:] = 0

        # Small baseline variation
        baseline = self.rng.normal(0, 0.005)
        signal += baseline

        # Low-level Gaussian noise
        noise = self.rng.normal(0, 0.005, signal.shape)
        signal += noise.astype(np.float32)

        return signal

    def __getitem__(self, index):
        batch_indices = []

        for cls in self.classes:
            selected = self.rng.choice(
                self.class_indices[cls],
                size=self.samples_per_class,
                replace=True
            )
            batch_indices.extend(selected)

        self.rng.shuffle(batch_indices)

        X_batch = self.X[batch_indices].copy()
        y_batch = self.y[batch_indices].copy()

        for i in range(len(X_batch)):
            X_batch[i] = self.augment(X_batch[i])

        return X_batch, y_batch


class MacroF1Callback(tf.keras.callbacks.Callback):

    def __init__(self, X_val, y_val, patience=10):
        super().__init__()

        self.X_val = X_val
        self.y_val = y_val
        self.patience = patience

        self.best_f1 = -1
        self.wait = 0

        self.model_path = MODEL_PATH

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(
            self.X_val,
            batch_size=512,
            verbose=0
        )

        predicted_classes = np.argmax(predictions, axis=1)

        score = f1_score(
            self.y_val,
            predicted_classes,
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0
        )

        print(f"\nValidation Macro F1: {score:.4f}")

        if score > self.best_f1:
            self.best_f1 = score
            self.wait = 0

            self.model.save(self.model_path)
            print("Saved new best model.")

        else:
            self.wait += 1

        if self.wait >= self.patience:
            print("\nEarly stopping: validation Macro F1 did not improve.")
            self.model.stop_training = True


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run prepare_data.py first."
        )

    data = np.load(DATA_PATH)

    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int32)

    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.int32)

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)

    print("\nOriginal training distribution:")
    for cls in range(4):
        print(f"Class {cls}: {np.sum(y_train == cls)}")

    train_generator = BalancedECGSequence(
        X_train,
        y_train,
        batch_size=BATCH_SIZE
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model(
        input_shape=X_train.shape[1:],
        num_classes=4
    )

    model.summary()

    callback = MacroF1Callback(
        X_val,
        y_val,
        patience=PATIENCE
    )

    model.fit(
        train_generator,
        epochs=EPOCHS,
        callbacks=[callback],
        verbose=1
    )

    print("\nTraining finished.")
    print(f"Best validation Macro F1: {callback.best_f1:.4f}")
    print(f"Best model saved at: {MODEL_PATH}")

    # Evaluate the saved best model on validation
    best_model = tf.keras.models.load_model(MODEL_PATH)

    predictions = best_model.predict(X_val, batch_size=512)
    predicted_classes = np.argmax(predictions, axis=1)

    print("\n========== VALIDATION REPORT ==========")

    print(classification_report(
        y_val,
        predicted_classes,
        labels=[0, 1, 2, 3],
        target_names=["N", "S", "V", "F"],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_val, predicted_classes))


if __name__ == "__main__":
    main()
