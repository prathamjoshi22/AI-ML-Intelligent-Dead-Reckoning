import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

data_folder = project_folder / "Data"

files = list(data_folder.rglob("S-S1.csv"))

if not files:
    raise FileNotFoundError("S-S1.csv not found")

file_path = files[0]

df = pd.read_csv(file_path, encoding="cp1252")

print("Dataset loaded successfully")
print()

print("Rows:", len(df))
print("Columns:", len(df.columns))

print()

print("Missing values:")
print(df.isnull().sum())

print()

print("Data types:")
print(df.dtypes)

print()

print("Basic statistics:")
print(df.describe())