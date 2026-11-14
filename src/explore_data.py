import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_record


# Load recording and annotations
record, annotation = load_record()

# Select the first ECG channel
signal = record.p_signal[:, 0]

# Sampling frequency
fs = record.fs

# Display the first 10 seconds
duration = 10
end_sample = duration * fs

# Generate time axis
time = np.arange(end_sample) / fs

# Select annotations inside the 10-second window
mask = annotation.sample < end_sample

samples = annotation.sample[mask]
symbols = np.array(annotation.symbol)[mask]

# Plot ECG
plt.figure(figsize=(15, 5))

plt.plot(time, signal[:end_sample], linewidth=1)

# Mark annotated positions
plt.scatter(
    samples / fs,
    signal[samples],
    color="red",
    s=25,
    label="Annotations"
)

# Add annotation labels
for sample, symbol in zip(samples, symbols):
    plt.annotate(
        symbol,
        (sample / fs, signal[sample]),
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=8
    )

plt.title("MIT-BIH ECG Recording 208 - First 10 Seconds")
plt.xlabel("Time (seconds)")
plt.ylabel("ECG Amplitude (mV)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()