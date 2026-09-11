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


time = pd.to_numeric(df[time_column], errors="coerce") / 1000
latitude = pd.to_numeric(df[lat_column], errors="coerce")
longitude = pd.to_numeric(df[lon_column], errors="coerce")
speed = pd.to_numeric(df[speed_column], errors="coerce")


start_time = 3562
end_time = 3622

mask = (time >= start_time) & (time <= end_time)

t = time[mask].to_numpy()
lat = latitude[mask].to_numpy()
lon = longitude[mask].to_numpy()
spd = speed[mask].to_numpy()


print("======================================")
print("       GPS INTERVAL VERIFICATION")
print("======================================")
print()

print(f"Interval: {start_time}s → {end_time}s")
print(f"Number of samples: {len(t)}")
print()

print("TIME")
print("First:", t[0])
print("Last :", t[-1])
print("Duration:", t[-1] - t[0])
print()

print("GPS SPEED")
print("Minimum:", spd.min())
print("Maximum:", spd.max())
print("Average:", spd.mean())
print()

print("GPS POSITION")
print("Starting latitude :", lat[0])
print("Starting longitude:", lon[0])
print()

print("Ending latitude   :", lat[-1])
print("Ending longitude  :", lon[-1])
print()


# Convert latitude/longitude differences to meters
lat_mean = np.mean(lat)

meters_per_degree_lat = 111320
meters_per_degree_lon = 111320 * np.cos(np.radians(lat_mean))

north = (lat - lat[0]) * meters_per_degree_lat
east = (lon - lon[0]) * meters_per_degree_lon

distance = np.sqrt(north**2 + east**2)

print("GPS DISPLACEMENT")
print("North displacement:", north[-1], "meters")
print("East displacement :", east[-1], "meters")
print("Straight-line displacement:", distance[-1], "meters")
print()

# Calculate distance between consecutive GPS positions
delta_north = np.diff(north)
delta_east = np.diff(east)

segment_distance = np.sqrt(
    delta_north**2 + delta_east**2
)

total_distance = np.sum(segment_distance)

print("GPS TRAVELED DISTANCE")
print("Total traveled distance:", total_distance, "meters")
print()

# Estimate distance from speed
speed_mps = spd / 3.6

dt = np.diff(t)

speed_distance = np.sum(
    (speed_mps[:-1] + speed_mps[1:]) / 2 * dt
)

print("SPEED-BASED DISTANCE")
print("Distance from GPS speed:", speed_distance, "meters")
print()

print("EXPECTED DISTANCE")
print(
    "Average speed × time:",
    spd.mean() / 3.6 * (t[-1] - t[0]),
    "meters"
)

print()

print("======================================")
print("        VERIFICATION COMPLETE")
print("======================================")