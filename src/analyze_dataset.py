
import os
from collections import Counter

import wfdb

DATA_DIR = "src/data/mitdb"

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

# AAMI class mapping
CLASS_MAPPING = {
    "N": ["N", "L", "R", "e", "j"],
    "S": ["A", "a", "J", "S"],
    "V": ["V", "E"],
    "F": ["F"]
}

class_counts = Counter()
ignored_counts = Counter()

# Convert symbols into their target classes
symbol_to_class = {
    symbol: class_name
    for class_name, symbols in CLASS_MAPPING.items()
    for symbol in symbols
}

for record_name in RECORDS:

    record_path = os.path.join(DATA_DIR, record_name)

    annotation = wfdb.rdann(record_path, "atr")

    for symbol in annotation.symbol:

        if symbol in symbol_to_class:
            class_name = symbol_to_class[symbol]
            class_counts[class_name] += 1

        else:
            ignored_counts[symbol] += 1

print("\n===== FOUR-CLASS DATASET =====")

for class_name in ["N", "S", "V", "F"]:
    print(f"{class_name}: {class_counts[class_name]}")

print("\nTotal heartbeats:", sum(class_counts.values()))

print("\n===== IGNORED ANNOTATIONS =====")

for symbol, count in ignored_counts.most_common():
    print(f"{symbol}: {count}")
