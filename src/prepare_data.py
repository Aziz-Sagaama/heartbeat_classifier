
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Load segmented heartbeat dataset
data = np.load("src/data/heartbeats.npz")

X = data["X"]
y = data["y"]
groups = data["groups"]

# Encode classes into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Class encoding:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# Split by recording, not by individual heartbeat
# Split by recording while balancing class distributions
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y_encoded, groups)
)



X_train = X[train_idx]
X_test = X[test_idx]

y_train = y_encoded[train_idx]
y_test = y_encoded[test_idx]

train_groups = groups[train_idx]
test_groups = groups[test_idx]

# Normalize each heartbeat independently
# Use training data statistics only
mean = X_train.mean(axis=1, keepdims=True)
std = X_train.std(axis=1, keepdims=True)

X_train = (X_train - mean) / (std + 1e-8)

test_mean = X_test.mean(axis=1, keepdims=True)
test_std = X_test.std(axis=1, keepdims=True)

X_test = (X_test - test_mean) / (test_std + 1e-8)

# Add channel dimension for 1D CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Save prepared datasets
np.savez_compressed(
    "src/data/prepared_data.npz",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    train_groups=train_groups,
    test_groups=test_groups,
    classes=encoder.classes_
)

print("\n===== DATA SPLIT RESULTS =====")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining classes:", Counter(y[train_idx]))
print("Testing classes:", Counter(y[test_idx]))

print("\nTraining recordings:", sorted(set(train_groups)))
print("Testing recordings:", sorted(set(test_groups)))

print("\nSaved: src/data/prepared_data.npz")
