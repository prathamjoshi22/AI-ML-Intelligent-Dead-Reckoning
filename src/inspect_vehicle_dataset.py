import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "Data"

files = list(data_folder.rglob("*.csv"))

print("==============================================")
print("          DATASET FILE INSPECTION")
print("==============================================")
print()

print("CSV files found:", len(files))
print()

for i, file in enumerate(files[:30], start=1):
    print(f"{i}. {file.relative_to(project_folder)}")

print()

print("==============================================")
print("          POSSIBLE VEHICLE FILES")
print("==============================================")
print()

for file in files:

    name = file.name.lower()

    if (
        "v-" in name
        or "vehicle" in name
        or "v_" in name
    ):

        print(file.relative_to(project_folder))

print()

print("==============================================")
print("          DATASET INSPECTION COMPLETE")
print("==============================================")