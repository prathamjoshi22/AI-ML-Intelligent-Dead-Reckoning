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
lat = pd.to_numeric(df[lat_column], errors="coerce")
lon = pd.to_numeric(df[lon_column], errors="coerce")
speed = pd.to_numeric(df[speed_column], errors="coerce")

data = pd.DataFrame({
    "time": time,
    "lat": lat,
    "lon": lon,
    "speed": speed
})

data = data.dropna().reset_index(drop=True)


lat_mean = data["lat"].mean()

meters_per_degree_lat = 111320
meters_per_degree_lon = (
    111320 * np.cos(np.radians(lat_mean))
)

data["north"] = (
    data["lat"] - data["lat"].iloc[0]
) * meters_per_degree_lat

data["east"] = (
    data["lon"] - data["lon"].iloc[0]
) * meters_per_degree_lon


# Distance between consecutive GPS positions
dn = np.diff(data["north"].to_numpy())
de = np.diff(data["east"].to_numpy())

segment_distance = np.sqrt(dn**2 + de**2)

data["segment_distance"] = np.concatenate(
    ([0], segment_distance)
)


# Search 60-second intervals
results = []

start_time = data["time"].min()
end_time = data["time"].max()

current_time = start_time

while current_time + 60 <= end_time:

    start = current_time
    finish = current_time + 60

    mask = (
        (data["time"] >= start) &
        (data["time"] <= finish)
    )

    interval = data[mask]

    if len(interval) < 300:
        current_time += 10
        continue

    t = interval["time"].to_numpy()
    spd = interval["speed"].to_numpy()

    north_values = interval["north"].to_numpy()
    east_values = interval["east"].to_numpy()

    # Position displacement
    dn_total = north_values[-1] - north_values[0]
    de_total = east_values[-1] - east_values[0]

    position_distance = np.sqrt(
        dn_total**2 + de_total**2
    )

    # GPS traveled distance
    traveled_distance = interval[
        "segment_distance"
    ].sum()

    # Speed-based distance
    speed_mps = spd / 3.6

    dt = np.diff(t)

    speed_distance = np.sum(
        (
            speed_mps[:-1] +
            speed_mps[1:]
        ) / 2 * dt
    )

    if speed_distance <= 0:
        current_time += 10
        continue

    consistency_error = abs(
        traveled_distance - speed_distance
    ) / speed_distance * 100

    average_speed = spd.mean()

    # Reject extremely slow intervals
    if average_speed < 8:
        current_time += 10
        continue

    results.append({
        "start": start,
        "end": finish,
        "average_speed": average_speed,
        "position_distance": position_distance,
        "traveled_distance": traveled_distance,
        "speed_distance": speed_distance,
        "consistency_error": consistency_error
    })

    current_time += 10


results = sorted(
    results,
    key=lambda x: x["consistency_error"]
)


print()
print("==============================================")
print("       VALID GNSS BLACKOUT SEARCH")
print("==============================================")
print()

print("Top 10 most consistent 60-second intervals:")
print()

for i, result in enumerate(results[:10], start=1):

    print(
        f"{i}. "
        f"{result['start']:.0f}s → "
        f"{result['end']:.0f}s"
    )

    print(
        f"   Average speed       : "
        f"{result['average_speed']:.2f} km/h"
    )

    print(
        f"   Position distance   : "
        f"{result['position_distance']:.2f} m"
    )

    print(
        f"   GPS traveled        : "
        f"{result['traveled_distance']:.2f} m"
    )

    print(
        f"   Speed-based distance: "
        f"{result['speed_distance']:.2f} m"
    )

    print(
        f"   Consistency error   : "
        f"{result['consistency_error']:.2f}%"
    )

    print()


print("==============================================")
print("SEARCH COMPLETED")
print("==============================================")