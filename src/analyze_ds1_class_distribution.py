
import numpy as np
import pandas as pd

data = np.load("src/data/processed/mitbih_dataset.npz")

y = data["y"].astype(str)
groups = data["groups"].astype(str)

DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230"
]

mask = np.isin(groups, DS1)

df = pd.DataFrame({
    "Patient": groups[mask],
    "Class": y[mask]
})

table = pd.crosstab(df["Patient"], df["Class"])

for cls in ["N", "S", "V", "F"]:
    if cls not in table.columns:
        table[cls] = 0

table = table[["N", "S", "V", "F"]]
table["Total_Abnormal"] = table[["S", "V", "F"]].sum(axis=1)

print("\n========== DS1 CLASS DISTRIBUTION ==========")
print(table.to_string())

print("\n========== TOTALS ==========")
print(table.sum())

print("\nNumber of patients containing S beats:",
      int((table["S"] > 0).sum()))
