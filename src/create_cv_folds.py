
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

DATA_PATH = Path("src/data/processed/mitbih_dataset.npz")
OUTPUT_PATH = Path("src/data/processed/ds1_cv_folds.npz")

DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230"
]

CLASS_TO_ID = {"N": 0, "S": 1, "V": 2, "F": 3}
CLASS_NAMES = ["N", "S", "V", "F"]

RANDOM_STATE = 42


def distribution(y):
    return np.bincount(y, minlength=4)


def main():
    data = np.load(DATA_PATH, allow_pickle=True)

    X = data["X"]
    raw_y = data["y"].astype(str)
    groups = data["groups"].astype(str)

    unknown = set(raw_y) - set(CLASS_TO_ID)
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")

    y = np.array([CLASS_TO_ID[label] for label in raw_y])

    mask = np.isin(groups, DS1)

    X_ds1 = X[mask]
    y_ds1 = y[mask]
    groups_ds1 = groups[mask]

    if set(groups_ds1) != set(DS1):
        raise ValueError("Some DS1 recordings are missing.")

    print("DS1 dataset:", X_ds1.shape)
    print("DS1 distribution:", distribution(y_ds1))

    # Isolate record 208 for a dedicated unseen-patient evaluation.
    record_208 = groups_ds1 == "208"
    idx_208 = np.where(record_208)[0]
    idx_other = np.where(~record_208)[0]

    folds = {}
    validation_coverage = np.zeros(len(y_ds1), dtype=int)

    # Fold 1: train on all other recordings, validate on record 208.
    fold_splits = [
        (idx_other, idx_208)
    ]

    # Folds 2-5: split the remaining 21 recordings into four groups.
    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for train_local, val_local in splitter.split(
        X_ds1[idx_other],
        y_ds1[idx_other],
        groups_ds1[idx_other]
    ):
        train_other = idx_other[train_local]
        val_other = idx_other[val_local]

        # Record 208 is included in training for these folds.
        train_idx = np.concatenate([train_other, idx_208])
        fold_splits.append((train_idx, val_other))

    for fold, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        train_groups = set(groups_ds1[train_idx])
        val_groups = set(groups_ds1[val_idx])

        assert train_groups.isdisjoint(val_groups)

        y_train = y_ds1[train_idx]
        y_val = y_ds1[val_idx]

        train_counts = distribution(y_train)
        val_counts = distribution(y_val)

        validation_coverage[val_idx] += 1

        folds[f"train_{fold}"] = train_idx
        folds[f"val_{fold}"] = val_idx

        print(f"\n========== FOLD {fold} ==========")
        print("Training examples:", len(train_idx))
        print("Validation examples:", len(val_idx))

        print("Training distribution:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {train_counts[i]}")

        print("Validation distribution:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {val_counts[i]}")

        print("Training recordings:", sorted(train_groups))
        print("Validation recordings:", sorted(val_groups))

    # Every DS1 heartbeat must be validated exactly once.
    assert np.all(validation_coverage == 1), \
        "Some samples are missing or appear multiple times in validation."

    np.savez_compressed(
        OUTPUT_PATH,
        X_ds1=X_ds1,
        y_ds1=y_ds1,
        groups_ds1=groups_ds1,
        **folds
    )

    print("\nAll samples validated exactly once.")
    print("Record 208 isolated in Fold 1.")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
