
import numpy as np

# MIT-BIH annotation symbols mapped to AAMI classes
CLASS_MAPPING = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "F": "F",
}


def extract_beats(record, annotation):

    # Explicitly select MLII instead of assuming channel 0
    channel_names = [
    name.strip().upper()
    for name in record.sig_name
]

    if "MLII" in channel_names:
        channel_index = channel_names.index("MLII")
    else:
        channel_index = 0
        print(
            f"WARNING: MLII unavailable. "
            f"Using {record.sig_name[channel_index]} instead."
        )

    signal = record.p_signal[:, channel_index]
    fs = record.fs

    before = int(0.25 * fs)
    after = int(0.40 * fs)

    beats = []
    labels = []

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol
    ):

        if symbol not in CLASS_MAPPING:
            continue

        start = sample - before
        end = sample + after

        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]

        if len(beat) != before + after:
            continue

        if not np.all(np.isfinite(beat)):
            continue

        beats.append(beat)
        labels.append(CLASS_MAPPING[symbol])

    X = np.asarray(beats, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(labels)

    return X, y
