import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold

# Load training data
data = np.load("src/data/prepared_data.npz")

X_train = data["X_train"]
y_train = data["y_train"]
train_groups = data["train_groups"]
classes = data["classes"]

# Create five folds
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Class mapping:", dict(enumerate(classes)))

# Inspect every possible validation fold
for fold, (fit_idx, val_idx) in enumerate(
    splitter.split(X_train, y_train, train_groups),
    start=1
):
    y_fit = y_train[fit_idx]
    y_val = y_train[val_idx]

    print(f"\n========== FOLD {fold} ==========")

    print("Training examples:", len(fit_idx))
    print("Validation examples:", len(val_idx))

    print("Training distribution:")
    for class_id, count in sorted(Counter(y_fit).items()):
        print(f"{classes[class_id]}: {count}")

    print("Validation distribution:")
    for class_id, count in sorted(Counter(y_val).items()):
        print(f"{classes[class_id]}: {count}")