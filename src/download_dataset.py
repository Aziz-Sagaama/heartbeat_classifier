
import os
import wfdb

# Where the recordings will be stored
DATA_DIR = "src/data/mitdb"

# The 48 MIT-BIH recordings
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

os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading MIT-BIH recordings...")
print("Number of recordings:", len(RECORDS))

wfdb.dl_database(
    db_dir="mitdb",
    dl_dir=DATA_DIR,
    records=RECORDS,
    annotators=["atr"]
)

print("\nDownload completed!")

# Verify downloaded recordings
downloaded = [
    r for r in RECORDS
    if os.path.exists(os.path.join(DATA_DIR, r + ".hea"))
    and os.path.exists(os.path.join(DATA_DIR, r + ".atr"))
]

print("Successfully downloaded:", len(downloaded), "/", len(RECORDS))
