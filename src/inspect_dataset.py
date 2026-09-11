import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

data_folder = project_folder / "Data"

files = list(data_folder.rglob("S-S1.csv"))

print("Searching for S-S1.csv...")
print()

if not files:
    print("S-S1.csv was not found.")
    print("Files found:")
    
    for file in data_folder.rglob("*"):
        if file.is_file():
            print(file)
    
    raise FileNotFoundError("S-S1.csv not found inside Data folder.")

file_path = files[0]

print("S-S1.csv found!")
print("Location:")
print(file_path)
print()

df = pd.read_csv(file_path, encoding="cp1252")

print("Dataset loaded successfully")
print()

print("Shape:")
print(df.shape)

print()

print("Columns:")
for i, col in enumerate(df.columns):
    print(i, "->", col)

print()

print("First 5 rows:")
print(df.head())