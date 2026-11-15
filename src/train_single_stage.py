
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback, EarlyStopping


# CONFIGURATION
DATA_PATH = Path("src/data/processed/ds1_cv_folds.npz")
MODEL_DIR = Path("models/single_stage")
OUTPUT_PATH = Path("src/data/processed/single_stage_oof.npz")

CLASS_NAMES = ["N", "S", "V", "F"]

BATCH_SIZE = 256
EPOCHS = 40
PATIENCE = 8
SEED = 42

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# CLASS WEIGHTS
def calculate_class_weights(y):

    counts = np.bincount(y, minlength=4)

    if np.any(counts == 0):
        raise ValueError(
            f"Missing training class. Counts: {counts}"
        )

    weights = np.sqrt(counts.max() / counts)
    weights = np.minimum(weights, 6.0)

    return {
        i: float(weights[i])
        for i in range(4)
    }


# CNN MODEL
def build_model(X_train):

    inputs = layers.Input(shape=X_train.shape[1:])

    # Statistics are learned from training data only
    normalization = layers.Normalization(axis=-1)
    normalization.adapt(X_train)

    x = normalization(inputs)

    x = layers.Conv1D(
        32, 7, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(
        64, 5, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(
        128, 3, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(4, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# VALIDATION MACRO F1
class MacroF1Callback(Callback):

    def __init__(self, X_val, y_val):
        super().__init__()

        self.X_val = X_val
        self.y_val = y_val

        self.best_f1 = -1
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):

        probabilities = self.model.predict(
            self.X_val,
            batch_size=512,
            verbose=0
        )

        predictions = np.argmax(probabilities, axis=1)

        score = f1_score(
            self.y_val,
            predictions,
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0
        )

        logs["val_macro_f1"] = score

        print(f" - val_macro_f1: {score:.4f}")

        if score > self.best_f1:
            self.best_f1 = score
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):

        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


# MAIN
def main():

    data = np.load(DATA_PATH, allow_pickle=True)

    X = data["X_ds1"].astype(np.float32)
    y = data["y_ds1"].astype(np.int32)
    groups = data["groups_ds1"]

    if X.ndim != 3 or X.shape[1:] != (234, 1):
        raise ValueError(f"Unexpected X shape: {X.shape}")

    if not np.all(np.isin(y, [0, 1, 2, 3])):
        raise ValueError("Unexpected class IDs.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    oof_prob = np.zeros((len(y), 4), dtype=np.float32)
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    validation_count = np.zeros(len(y), dtype=np.int32)

    fold_scores = []

    for fold in range(1, 6):

        print(f"\n{'=' * 20} FOLD {fold}/5 {'=' * 20}")

        train_idx = data[f"train_{fold}"]
        val_idx = data[f"val_{fold}"]

        # Check patient-level separation
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])

        if train_groups.intersection(val_groups):
            raise ValueError(
                f"Patient leakage detected in fold {fold}"
            )

        validation_count[val_idx] += 1

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        print("Training:", len(y_train))
        print("Validation:", len(y_val))

        print(
            "Training distribution:",
            np.bincount(y_train, minlength=4)
        )

        print(
            "Validation distribution:",
            np.bincount(y_val, minlength=4)
        )

        class_weights = calculate_class_weights(y_train)
        print("Class weights:", class_weights)

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED + fold)

        model = build_model(X_train)

        f1_callback = MacroF1Callback(X_val, y_val)

        early_stopping = EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            class_weight=class_weights,
            validation_data=(X_val, y_val),
            callbacks=[f1_callback, early_stopping],
            verbose=1
        )

        probabilities = model.predict(
            X_val,
            batch_size=512,
            verbose=0
        )

        predictions = np.argmax(probabilities, axis=1)

        oof_prob[val_idx] = probabilities
        oof_pred[val_idx] = predictions

        fold_f1 = f1_score(
            y_val,
            predictions,
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0
        )

        fold_scores.append(fold_f1)

        print(f"\nFold {fold} macro F1: {fold_f1:.4f}")

        model.save(MODEL_DIR / f"fold_{fold}.keras")

        del model
        tf.keras.backend.clear_session()

    # VERIFY OOF COVERAGE
    if not np.all(validation_count == 1):
        raise ValueError(
            "Each sample must appear in validation exactly once."
        )

    if np.any(oof_pred < 0):
        raise ValueError("Some samples have no OOF prediction.")

    # FINAL EVALUATION
    print("\n========== SINGLE-STAGE OOF RESULTS ==========")

    print(classification_report(
        y,
        oof_pred,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y, oof_pred, labels=[0, 1, 2, 3]))

    overall_f1 = f1_score(
        y,
        oof_pred,
        labels=[0, 1, 2, 3],
        average="macro",
        zero_division=0
    )

    print(f"\nOverall OOF macro F1: {overall_f1:.4f}")
    print(f"Mean fold macro F1: {np.mean(fold_scores):.4f}")

    np.savez_compressed(
        OUTPUT_PATH,
        y_true=y,
        y_pred=oof_pred,
        y_prob=oof_prob,
        groups=groups
    )

    print("\nSaved predictions:", OUTPUT_PATH)
    print("Saved models:", MODEL_DIR)


if __name__ == "__main__":
    main()
