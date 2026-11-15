
import os
import wfdb
import numpy as np

from pathlib import Path
from collections import Counter
from preprocessing import extract_beats


# MIT-BIH Arrhythmia Database: 48 recordings
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

# Paths relative to this Python file
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "mitdb"
OUTPUT_DIR = BASE_DIR / "data" / "processed"


def download_dataset():

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading MIT-BIH recordings...")

    wfdb.dl_database(
        "mitdb",
        dl_dir=str(DATA_DIR),
        records=RECORDS
    )

    print("Download completed!")


def build_dataset():

    all_beats = []
    all_labels = []
    all_record_ids = []

    for index, record_name in enumerate(RECORDS, start=1):

        print(f"\nProcessing record {record_name} "
              f"({index}/{len(RECORDS)})")

        record_path = DATA_DIR / record_name

        # Load ECG signal and expert annotations
        record = wfdb.rdrecord(str(record_path))
        annotation = wfdb.rdann(str(record_path), "atr")

        # Extract individual heartbeats
        X, y = extract_beats(record, annotation)

        # Preserve the source recording for every heartbeat
        record_ids = np.full(len(y), record_name)

        all_beats.append(X)
        all_labels.append(y)
        all_record_ids.append(record_ids)

        print(f"Extracted {len(y)} heartbeats")

    # Combine all recordings
    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_record_ids, axis=0)

    # Save dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "mitbih_dataset.npz"

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        groups=groups
    )

    # Display results
    print("\n========== COMPLETE DATASET ==========")

    print("Heartbeat dataset shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Recording IDs shape:", groups.shape)

    print("\nClass distribution:")

    for label, count in sorted(Counter(y).items()):
        print(f"{label}: {count}")

    print("\nNumber of recordings:", len(np.unique(groups)))

    print("\nDataset saved at:", output_path)


if __name__ == "__main__":

    download_dataset()
    build_dataset()
