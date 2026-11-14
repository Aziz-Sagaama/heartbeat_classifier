import wfdb
import os

# Dataset configuration
DATABASE = "mitdb"
RECORD_NAME = "208"
DATA_DIR = "src/data/mitdb"


def download_record():

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading ECG recording...")

    wfdb.dl_database(
        db_dir=DATABASE,
        dl_dir=DATA_DIR,
        records=[RECORD_NAME]
    )

    print("Download completed!")


def load_record():

    record_path = os.path.join(DATA_DIR, RECORD_NAME)

    # Load ECG signal
    record = wfdb.rdrecord(record_path)

    # Load expert annotations
    annotation = wfdb.rdann(record_path, "atr")

    return record, annotation


if __name__ == "__main__":

    download_record()

    record, annotation = load_record()

    print("\n===== ECG RECORD INFORMATION =====")

    print("Sampling frequency:", record.fs)
    print("Number of channels:", record.n_sig)
    print("Channel names:", record.sig_name)
    print("Signal shape:", record.p_signal.shape)

    print("\n===== ANNOTATION INFORMATION =====")

    print("Number of annotations:", len(annotation.sample))
    print("First 10 annotation positions:", annotation.sample[:10])
    print("First 10 annotation symbols:", annotation.symbol[:10])