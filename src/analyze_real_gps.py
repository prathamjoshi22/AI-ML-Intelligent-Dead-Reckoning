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
heading_column = find_column("GPS ORIENTATION")


time = pd.to_numeric(df[time_column], errors="coerce") / 1000
lat = pd.to_numeric(df[lat_column], errors="coerce")
lon = pd.to_numeric(df[lon_column], errors="coerce")
speed = pd.to_numeric(df[speed_column], errors="coerce")
heading = pd.to_numeric(df[heading_column], errors="coerce")


gps = pd.DataFrame({
    "time": time,
    "lat": lat,
    "lon": lon,
    "speed": speed,
    "heading": heading
})


# Keep only rows where GPS position actually changes
gps["lat_change"] = gps["lat"].diff().abs()
gps["lon_change"] = gps["lon"].diff().abs()

gps_updates = gps[
    (gps["lat_change"] > 1e-10) |
    (gps["lon_change"] > 1e-10)
].copy()

gps_updates = gps_updates.reset_index(drop=True)


print("==============================================")
print("          REAL GPS UPDATE ANALYSIS")
print("==============================================")
print()

print("Total sensor rows:", len(df))
print("Actual GPS position updates:", len(gps_updates))
print()

print("Approximate GPS update frequency:")

time_difference = gps_updates["time"].diff().dropna()

print("Average interval:",
      time_difference.mean(), "seconds")

print("Minimum interval:",
      time_difference.min(), "seconds")

print("Maximum interval:",
      time_difference.max(), "seconds")

print()

print("First 20 REAL GPS updates:")
print()

for i in range(min(20, len(gps_updates))):

    row = gps_updates.iloc[i]

    print(
        f"{i:02d} | "
        f"time={row['time']:.3f}s | "
        f"lat={row['lat']:.6f} | "
        f"lon={row['lon']:.6f} | "
        f"speed={row['speed']:.2f} km/h | "
        f"heading={row['heading']:.2f}"
    )

print()

print("==============================================")
print("       GPS UPDATE ANALYSIS COMPLETE")
print("==============================================")