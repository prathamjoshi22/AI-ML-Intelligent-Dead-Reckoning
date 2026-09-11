import pandas as pd
import numpy as np
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
lat_column = find_column("GPS LATITUDE")
lon_column = find_column("GPS LONGITUDE")
speed_column = find_column("GPS SPEED")
orientation_column = find_column("GPS ORIENTATION")


time = pd.to_numeric(df[time_column], errors="coerce") / 1000
lat = pd.to_numeric(df[lat_column], errors="coerce")
lon = pd.to_numeric(df[lon_column], errors="coerce")
speed = pd.to_numeric(df[speed_column], errors="coerce")
orientation = pd.to_numeric(df[orientation_column], errors="coerce")


print("======================================")
print("       GPS UPDATE INSPECTION")
print("======================================")
print()

print("Total rows:", len(df))
print()

print("First 30 GPS values:")
print()

for i in range(30):
    print(
        f"{i:02d} | "
        f"time={time.iloc[i]:.3f}s | "
        f"lat={lat.iloc[i]:.6f} | "
        f"lon={lon.iloc[i]:.6f} | "
        f"speed={speed.iloc[i]:.2f} | "
        f"heading={orientation.iloc[i]:.2f}"
    )

print()
print("======================================")
print("CHECKING GPS POSITION CHANGES")
print("======================================")
print()

lat_change = lat.diff().abs()
lon_change = lon.diff().abs()

position_changed = (
    (lat_change > 1e-10) |
    (lon_change > 1e-10)
)

print(
    "Rows where GPS position changed:",
    position_changed.sum()
)

print(
    "Percentage:",
    position_changed.mean() * 100
)

print()

print("First 30 rows with GPS position changes:")
print()

changed_indices = np.where(position_changed.to_numpy())[0]

for i in changed_indices[:30]:

    print(
        f"row={i} | "
        f"time={time.iloc[i]:.3f}s | "
        f"lat={lat.iloc[i]:.6f} | "
        f"lon={lon.iloc[i]:.6f}"
    )

print()
print("======================================")
print("CHECKING TIME DIFFERENCES")
print("======================================")
print()

time_diff = time.diff().dropna()

print("Minimum dt:", time_diff.min())
print("Maximum dt:", time_diff.max())
print("Average dt:", time_diff.mean())

print()

print("======================================")
print("INSPECTION COMPLETE")
print("======================================")