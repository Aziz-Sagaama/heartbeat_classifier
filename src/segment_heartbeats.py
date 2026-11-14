
import os
import numpy as np
import wfdb
from collections import Counter

DATA_DIR = "src/data/mitdb"
OUTPUT_DIR = "src/data"

RECORDS = [
    "100", "101", "102", "103", "104", "105",
    "106", "107", "108", "109", "111", "112",
    "113", "114", "115", "116", "117", "118",
    "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215",
    "217", "219", "220", "221", "222", "223",
    "228", "230", "231", "232", "233", "234"
]

CLASS_MAPPING = {
    "N": ["N", "L", "R", "e", "j"],
    "S": ["A", "a", "J", "S"],
    "V": ["V", "E"],
    "F": ["F"]
}

symbol_to_class = {
    symbol: class_name
    for class_name, symbols in CLASS_MAPPING.items()
    for symbol in symbols
}

# 0.2 seconds before and 0.4 seconds after
BEFORE = 72
AFTER = 144
WINDOW_SIZE = BEFORE + AFTER

X = []
y = []
groups = []

os.makedirs(OUTPUT_DIR, exist_ok=True)

for record_name in RECORDS:

    print("Processing recording:", record_name)

    record_path = os.path.join(DATA_DIR, record_name)

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    # Use MLII (first ECG channel)
    signal = record.p_signal[:, 0]

    for sample, symbol in zip(annotation.sample, annotation.symbol):

        if symbol not in symbol_to_class:
            continue

        start = sample - BEFORE
        end = sample + AFTER

        # Avoid segments outside signal boundaries
        if start < 0 or end > len(signal):
            continue

        heartbeat = signal[start:end]

        X.append(heartbeat)
        y.append(symbol_to_class[symbol])
        groups.append(record_name)

# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y)
groups = np.array(groups)

# Save dataset
output_path = os.path.join(OUTPUT_DIR, "heartbeats.npz")

np.savez_compressed(
    output_path,
    X=X,
    y=y,
    groups=groups
)

print("\n===== SEGMENTATION RESULTS =====")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of recordings:", len(np.unique(groups)))
print("Class distribution:", Counter(y))
print("Saved to:", output_path)
