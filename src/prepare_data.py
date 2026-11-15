
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = Path("src/data/processed")
INPUT_FILE = DATA_DIR / "mitbih_dataset.npz"

DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230"
]

DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234"
]

CLASS_TO_ID = {"N": 0, "S": 1, "V": 2, "F": 3}
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def show_distribution(name, y):
    print(f"\n{name}: {len(y)} beats")
    for cls in range(4):
        count = np.sum(y == cls)
        print(f"{['N', 'S', 'V', 'F'][cls]}: {count}")


def find_best_split(X, y, groups):
    """Find a group-independent split with good class representation."""
    splitter = GroupShuffleSplit(
        n_splits=3000,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    total_counts = np.bincount(y, minlength=4)
    best_score = float("inf")
    best_indices = None

    for train_idx, val_idx in splitter.split(X, y, groups):
        train_counts = np.bincount(y[train_idx], minlength=4)
        val_counts = np.bincount(y[val_idx], minlength=4)

        if np.any(train_counts == 0) or np.any(val_counts[1:] < 10):
            continue

        # Each class should have approximately 20% in validation.
        ratios = val_counts / total_counts
        score = np.mean(np.abs(ratios - VALIDATION_SIZE))

        if score < best_score:
            best_score = score
            best_indices = (train_idx, val_idx)

    if best_indices is None:
        raise ValueError(
            "Could not find a suitable validation split with at least "
            "10 examples of each minority class. Check class distribution "
            "across DS1 recordings."
        )

    return best_indices




def main():
    data = np.load(INPUT_FILE, allow_pickle=True)

    X = data["X"]
    raw_y = data["y"].astype(str)
    groups = data["groups"].astype(str)

    unknown = set(raw_y) - set(CLASS_TO_ID)
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")

    y = np.array([CLASS_TO_ID[label] for label in raw_y])

    print("Original dataset:", X.shape)
    show_distribution("Original distribution", y)

    ds1_mask = np.isin(groups, DS1)
    ds2_mask = np.isin(groups, DS2)

    X_ds1, y_ds1 = X[ds1_mask], y[ds1_mask]
    groups_ds1 = groups[ds1_mask]

    X_test, y_test = X[ds2_mask], y[ds2_mask]

    excluded_records = sorted(set(groups) - set(DS1) - set(DS2))

    print("Excluded recordings:", excluded_records)
    print("DS1 examples:", len(X_ds1))
    print("DS2 examples:", len(X_test))

    train_idx, val_idx = find_best_split(
        X_ds1, y_ds1, groups_ds1
    )

    X_train, y_train = X_ds1[train_idx], y_ds1[train_idx]
    X_val, y_val = X_ds1[val_idx], y_ds1[val_idx]

    

    train_records = set(groups_ds1[train_idx])
    val_records = set(groups_ds1[val_idx])
    test_records = set(groups[ds2_mask])

    assert train_records.isdisjoint(val_records)
    assert train_records.isdisjoint(test_records)
    assert val_records.isdisjoint(test_records)

    show_distribution("Balanced training", y_train)
    show_distribution("Validation", y_val)
    show_distribution("Test", y_test)

    print("\nTraining records:", sorted(train_records))
    print("Validation records:", sorted(val_records))
    print("Test records:", sorted(test_records))

    np.savez_compressed(
        DATA_DIR / "final_splits.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test
    )

    print("\nSaved:", DATA_DIR / "final_splits.npz")


if __name__ == "__main__":
    main()
