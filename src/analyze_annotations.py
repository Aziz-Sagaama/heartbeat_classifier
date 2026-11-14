
from collections import Counter
from data_loader import load_record

record, annotation = load_record()

counts = Counter(annotation.symbol)

print("===== ANNOTATION DISTRIBUTION =====")

for symbol, count in counts.most_common():
    print(f"{symbol}: {count}")
