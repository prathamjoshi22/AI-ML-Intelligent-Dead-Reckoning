import pandas as pd
from pathlib import Path


project_folder = Path(__file__).resolve().parent.parent

data_folder = project_folder / "Data"

files = list(data_folder.rglob("S-S1.csv"))

if not files:
    raise FileNotFoundError("S-S1.csv was not found.")

file_path = files[0]

df = pd.read_csv(file_path, encoding="cp1252")

df.columns = df.columns.str.strip()


def find_column(keyword):
    for column in df.columns:
        if keyword.lower() in column.lower():
            return column
    return None


time_column = find_column("TIME SINCE START")
speed_column = find_column("GPS SPEED")
latitude_column = find_column("GPS LATITUDE")
longitude_column = find_column("GPS LONGITUDE")


print("======================================")
print("        IO-VNBD S-S1 DATA INFO")
print("======================================")
print()

print("Rows:", len(df))
print("Columns:", len(df.columns))
print()


if time_column:
    time = pd.to_numeric(df[time_column], errors="coerce") / 1000

    print("Recording duration:")
    print(f"{time.min():.2f} seconds → {time.max():.2f} seconds")
    print(f"Total duration: {time.max() - time.min():.2f} seconds")
    print()


if speed_column:
    speed = pd.to_numeric(df[speed_column], errors="coerce")

    print("GPS SPEED")
    print("Minimum:", speed.min())
    print("Maximum:", speed.max())
    print("Average:", speed.mean())
    print()


if latitude_column and longitude_column:

    latitude = pd.to_numeric(
        df[latitude_column],
        errors="coerce"
    )

    longitude = pd.to_numeric(
        df[longitude_column],
        errors="coerce"
    )

    print("GPS POSITION")
    print("Latitude range:")
    print(latitude.min(), "→", latitude.max())

    print("Longitude range:")
    print(longitude.min(), "→", longitude.max())
    print()


print("Missing values:")
print(df.isnull().sum())

print()
print("======================================")
print("Analysis complete")
print("======================================")