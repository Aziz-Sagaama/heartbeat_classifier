
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# Standard DS1 records
DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230"
]

# Load original dataset
data = np.load("src/data/processed/mitbih_dataset.npz")

X_all = data["X"]
y_all = data["y"]
groups_all = data["groups"].astype(str)

# Keep only DS1 patients
mask = np.isin(groups_all, DS1)

X = X_all[mask]
y_original = y_all[mask]
groups = groups_all[mask]

# Binary labels: N = 0, S/V/F = 1
y = (y_original != "N").astype(np.int32)

print("DS1 dataset:", X.shape)
print("Normal:", np.sum(y == 0))
print("Abnormal:", np.sum(y == 1))

# Generate patient-level stratified folds
skf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

folds = {}
validation_counts = np.zeros(len(y), dtype=np.int32)

for fold, (train_idx, val_idx) in enumerate(
    skf.split(X, y, groups), start=1
):
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

    # Check patient leakage
    assert train_groups.isdisjoint(val_groups)

    validation_counts[val_idx] += 1

    folds[f"train_idx_{fold}"] = train_idx
    folds[f"val_idx_{fold}"] = val_idx

    print(f"\n========== FOLD {fold} ==========")
    print("Training examples:", len(train_idx))
    print("Validation examples:", len(val_idx))
    print("Training patients:", sorted(train_groups))
    print("Validation patients:", sorted(val_groups))
    print("Train distribution:", np.bincount(y[train_idx], minlength=2))
    print("Validation distribution:", np.bincount(y[val_idx], minlength=2))

# Every DS1 beat must be validated exactly once
assert np.all(validation_counts == 1)

# Save dataset and fold indices
folds["X"] = X
folds["y"] = y
folds["groups"] = groups

np.savez_compressed(
    "src/data/processed/binary_cv_folds.npz",
    **folds
)

print("\nAll checks passed.")
print("Every DS1 beat appears in validation exactly once.")
print("No patient overlap between training and validation.")
print("Saved: src/data/processed/binary_cv_folds.npz")
