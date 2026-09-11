import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "Data"

files = list(data_folder.rglob("V-S1.csv"))

if not files:
    raise FileNotFoundError("V-S1.csv was not found.")

file_path = files[0]

print("==============================================")
print("        VEHICLE REFERENCE INSPECTION")
print("==============================================")
print()

print("File:")
print(file_path)
print()

df = pd.read_csv(file_path, encoding="cp1252")

df.columns = df.columns.str.strip()

print("Rows:", len(df))
print("Columns:", len(df.columns))
print()

print("COLUMN NAMES")
print("----------------------------------------------")

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print()

print("==============================================")
print("             FIRST 5 ROWS")
print("==============================================")
print()

print(df.head())

print()

print("==============================================")
print("             DATA TYPES")
print("==============================================")
print()

print(df.dtypes)

print()

print("==============================================")
print("          MISSING VALUES")
print("==============================================")
print()

missing = df.isnull().sum()

for column in df.columns:
    if missing[column] > 0:
        print(f"{column}: {missing[column]}")

print()

print("==============================================")
print("       VEHICLE REFERENCE INSPECTION DONE")
print("==============================================")