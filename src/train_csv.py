import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras import layers, models, callbacks

FOLDS_PATH = Path("src/data/processed/ds1_cv_folds.npz")
OUTPUT_PATH = Path("src/data/processed/oof_predictions.npz")
MODEL_DIR = Path("models/cv")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["N", "S", "V", "F"]
NUM_CLASSES = 4
BATCH_SIZE = 128
EPOCHS = 20
PATIENCE = 5
SEED = 42


def build_model():
    model = models.Sequential([
        layers.Input(shape=(234, 1)),

        layers.Conv1D(32, 7, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(64, 5, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, 3, activation="relu"),
        layers.BatchNormalization(),

        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


class BalancedSequence(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=128):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.rng = np.random.default_rng(SEED)

        self.class_indices = [
            np.where(y == c)[0] for c in range(NUM_CLASSES)
        ]

        if any(len(indices) == 0 for indices in self.class_indices):
            raise ValueError("A training fold is missing a class.")

        self.steps = int(np.ceil(len(y) / batch_size))

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        per_class = self.batch_size // NUM_CLASSES
        indices = np.concatenate([
            self.rng.choice(c_indices, per_class, replace=True)
            for c_indices in self.class_indices
        ])
        self.rng.shuffle(indices)

        X_batch = self.X[indices].copy()
        y_batch = self.y[indices]

        # Mild ECG augmentation
        scales = self.rng.uniform(
            0.90, 1.10, size=(len(indices), 1, 1)
        )
        noise = self.rng.normal(
            0, 0.005, size=X_batch.shape
        )
        baseline = self.rng.normal(
            0, 0.005, size=(len(indices), 1, 1)
        )

        X_batch = X_batch * scales + noise + baseline

        return X_batch.astype(np.float32), y_batch


class MacroF1Callback(callbacks.Callback):
    def __init__(self, X_val, y_val, fold):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.fold = fold
        self.best_f1 = -1
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(
            self.X_val, batch_size=512, verbose=0
        )
        predictions = np.argmax(probabilities, axis=1)

        score = f1_score(
            self.y_val,
            predictions,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0
        )

        print(f" - val_macro_f1: {score:.4f}")

        if score > self.best_f1:
            self.best_f1 = score
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= PATIENCE:
            print(f"Early stopping. Best macro F1: {self.best_f1:.4f}")
            self.model.stop_training = True


def main():
    tf.keras.utils.set_random_seed(SEED)

    data = np.load(FOLDS_PATH, allow_pickle=True)

    X = data["X_ds1"].astype(np.float32)
    y = data["y_ds1"].astype(np.int32)
    groups = data["groups_ds1"]

    oof_probabilities = np.full(
        (len(y), NUM_CLASSES), np.nan, dtype=np.float32
    )

    for fold in range(1, 6):
        print(f"\n{'=' * 20} FOLD {fold}/5 {'=' * 20}")

        train_idx = data[f"train_{fold}"]
        val_idx = data[f"val_{fold}"]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        print("Train:", X_train.shape)
        print("Validation:", X_val.shape)
        print("Training class counts:", np.bincount(y_train, minlength=4))
        print("Validation class counts:", np.bincount(y_val, minlength=4))

        train_sequence = BalancedSequence(
            X_train, y_train, BATCH_SIZE
        )

        model = build_model()
        f1_callback = MacroF1Callback(X_val, y_val, fold)

        model.fit(
            train_sequence,
            epochs=EPOCHS,
            callbacks=[f1_callback],
            verbose=1
        )

        if f1_callback.best_weights is None:
            raise RuntimeError(f"No best weights saved for fold {fold}")

        model.set_weights(f1_callback.best_weights)

        model_path = MODEL_DIR / f"fold_{fold}.keras"
        model.save(model_path)

        probabilities = model.predict(
            X_val, batch_size=512, verbose=0
        )
        oof_probabilities[val_idx] = probabilities

        print(f"Fold {fold} best validation macro F1:",
              f"{f1_callback.best_f1:.4f}")

        del model
        tf.keras.backend.clear_session()

    if np.isnan(oof_probabilities).any():
        raise RuntimeError("Some DS1 samples have no OOF predictions.")

    oof_predictions = np.argmax(oof_probabilities, axis=1)

    print("\n========== COMBINED OOF RESULTS ==========")
    print(classification_report(
        y,
        oof_predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y, oof_predictions))

    print("Overall OOF macro F1:",
          f1_score(y, oof_predictions, average="macro",
                   zero_division=0))

    np.savez_compressed(
        OUTPUT_PATH,
        y_true=y,
        y_pred=oof_predictions,
        y_prob=oof_probabilities,
        groups=groups
    )

    print("\nOOF predictions saved to:", OUTPUT_PATH)
    print("Fold models saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()