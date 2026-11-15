from __future__ import annotations

import csv
import io

import numpy as np


def parse_csv_signal(content: bytes) -> np.ndarray:
    try:
        text = content.decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))
        rows = [[cell.strip() for cell in row if cell.strip()] for row in rows]
        rows = [row for row in rows if row]

        # Accept the common exports: one numeric column, a header, or
        # an index/value pair produced by a dataframe export.
        if rows and len(rows[0]) == 2:
            data_rows = rows
            try:
                float(data_rows[0][0])
                float(data_rows[0][1])
            except ValueError:
                data_rows = data_rows[1:]
            if all(len(row) == 2 for row in data_rows):
                values = [row[1] for row in data_rows]
            else:
                values = [cell for row in rows for cell in row]
        else:
            values = [cell for row in rows for cell in row]
            try:
                [float(value) for value in values]
            except ValueError:
                if values:
                    values = values[1:]

        signal = np.asarray([float(value) for value in values], dtype=np.float32)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("CSV must contain numeric ECG values only.") from exc

    if signal.size != 234:
        raise ValueError(f"Expected exactly 234 ECG values, received {signal.size}.")
    if not np.isfinite(signal).all():
        raise ValueError("ECG values must be finite numbers.")
    return signal
