
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import Sequence


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path("src/data/processed/ds1_cv_folds.npz")
MODEL_DIR = Path("models/two_stage")
OUTPUT_PATH = Path("src/data/processed/two_stage_oof.npz")

CLASS_NAMES = ["N", "S", "V", "F"]

BATCH_SIZE = 128
EPOCHS = 25
PATIENCE = 6
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# --------------------------------------------------
# BALANCED BATCH GENERATOR
# --------------------------------------------------

class BalancedSequence(Sequence):

    def __init__(self, X, y, n_classes, batch_size=128):
        super().__init__()

        self.X = X
        self.y = y
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.per_class = batch_size // n_classes

        self.indices = [
            np.where(y == c)[0]
            for c in range(n_classes)
        ]

        if any(len(idx) == 0 for idx in self.indices):
            raise ValueError("A training class has no examples.")

        self.steps = max(1, int(np.ceil(len(y) / batch_size)))

    def __len__(self):
        return self.steps

    def __getitem__(self, index):

        selected = []

        for class_indices in self.indices:
            selected.extend(
                np.random.choice(
                    class_indices,
                    self.per_class,
                    replace=True
                )
            )

        selected = np.array(selected)
        np.random.shuffle(selected)

        X_batch = self.X[selected].copy()
        y_batch = self.y[selected]

        # Mild ECG augmentation
        for i in range(len(X_batch)):

            signal = X_batch[i, :, 0]

            # Amplitude scaling
            signal *= np.random.uniform(0.90, 1.10)

            # Small temporal shift
            shift = np.random.randint(-5, 6)

            if shift > 0:
                signal[shift:] = signal[:-shift].copy()
                signal[:shift] = 0

            elif shift < 0:
                signal[:shift] = signal[-shift:].copy()
                signal[shift:] = 0

            # Baseline variation and Gaussian noise
            signal += np.random.normal(0, 0.005)
            signal += np.random.normal(
                0, 0.005, size=signal.shape
            )

        return X_batch, y_batch


# --------------------------------------------------
# CNN ARCHITECTURE
# --------------------------------------------------

def build_model(input_shape, n_classes):

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(
        32, 7, padding="same", activation="relu"
    )(inputs)
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

    outputs = layers.Dense(
        n_classes, activation="softmax"
    )(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# MACRO F1 CALLBACK
# --------------------------------------------------

class MacroF1Callback(Callback):

    def __init__(self, X_val, y_val, n_classes):
        super().__init__()

        self.X_val = X_val
        self.y_val = y_val
        self.n_classes = n_classes

        self.best_f1 = -1
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):

        probabilities = self.model.predict(
            self.X_val, verbose=0
        )

        predictions = np.argmax(probabilities, axis=1)

        score = f1_score(
            self.y_val,
            predictions,
            labels=list(range(self.n_classes)),
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


# --------------------------------------------------
# TRAIN ONE MODEL
# --------------------------------------------------

def train_model(X_train, y_train, X_val, y_val, n_classes):

    generator = BalancedSequence(
        X_train,
        y_train,
        n_classes,
        BATCH_SIZE
    )

    model = build_model(
        X_train.shape[1:],
        n_classes
    )

    f1_callback = MacroF1Callback(
        X_val,
        y_val,
        n_classes
    )

    early_stopping = EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=PATIENCE,
        restore_best_weights=True
    )

    model.fit(
        generator,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[f1_callback, early_stopping],
        verbose=1
    )

    return model, f1_callback.best_f1


# --------------------------------------------------
# MAIN CROSS-VALIDATION
# --------------------------------------------------

def main():

    data = np.load(DATA_PATH, allow_pickle=True)

    X = data["X_ds1"].astype(np.float32)
    y = data["y_ds1"].astype(np.int32)
    groups = data["groups_ds1"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    oof_prob = np.zeros((len(y), 4), dtype=np.float32)
    oof_pred = np.zeros(len(y), dtype=np.int64)

    fold_scores = []

    for fold in range(1, 6):

        print(f"\n{'=' * 20} FOLD {fold}/5 {'=' * 20}")

        train_idx = data[f"train_{fold}"]
        val_idx = data[f"val_{fold}"]

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        # ------------------------------------------
        # STAGE 1: NORMAL VS ABNORMAL
        # N = 0, S/V/F = 1
        # ------------------------------------------

        print("\nSTAGE 1: NORMAL VS ABNORMAL")

        y_train_binary = (y_train != 0).astype(np.int32)
        y_val_binary = (y_val != 0).astype(np.int32)

        model_binary, score_binary = train_model(
            X_train,
            y_train_binary,
            X_val,
            y_val_binary,
            n_classes=2
        )

        # ------------------------------------------
        # STAGE 2: S VS V VS F
        # Train ONLY on abnormal beats
        # ------------------------------------------

        print("\nSTAGE 2: S VS V VS F")

        train_abnormal = y_train != 0
        val_abnormal = y_val != 0

        X_train_abnormal = X_train[train_abnormal]
        X_val_abnormal = X_val[val_abnormal]

        # S=0, V=1, F=2
        y_train_abnormal = y_train[train_abnormal] - 1
        y_val_abnormal = y_val[val_abnormal] - 1

        model_abnormal, score_abnormal = train_model(
            X_train_abnormal,
            y_train_abnormal,
            X_val_abnormal,
            y_val_abnormal,
            n_classes=3
        )

        # ------------------------------------------
        # COMBINE BOTH MODELS
        # ------------------------------------------

        p_binary = model_binary.predict(
            X_val, verbose=0
        )

        p_abnormal = model_abnormal.predict(
            X_val, verbose=0
        )

        combined = np.zeros((len(X_val), 4), dtype=np.float32)

        combined[:, 0] = p_binary[:, 0]

        combined[:, 1] = p_binary[:, 1] * p_abnormal[:, 0]
        combined[:, 2] = p_binary[:, 1] * p_abnormal[:, 1]
        combined[:, 3] = p_binary[:, 1] * p_abnormal[:, 2]

        oof_prob[val_idx] = combined
        oof_pred[val_idx] = np.argmax(combined, axis=1)

        fold_f1 = f1_score(
            y_val,
            oof_pred[val_idx],
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0
        )

        fold_scores.append(fold_f1)

        print(f"\nFold {fold} binary macro F1: {score_binary:.4f}")
        print(f"Fold {fold} abnormal macro F1: {score_abnormal:.4f}")
        print(f"Fold {fold} combined macro F1: {fold_f1:.4f}")

        model_binary.save(
            MODEL_DIR / f"fold_{fold}_binary.keras"
        )

        model_abnormal.save(
            MODEL_DIR / f"fold_{fold}_abnormal.keras"
        )

        del model_binary, model_abnormal
        tf.keras.backend.clear_session()

    # ----------------------------------------------
    # FINAL OOF EVALUATION
    # ----------------------------------------------

    print("\n========== TWO-STAGE OOF RESULTS ==========")

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
