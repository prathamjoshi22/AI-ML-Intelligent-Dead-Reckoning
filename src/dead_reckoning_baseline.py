import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
latitude_column = find_column("GPS LATITUDE")
longitude_column = find_column("GPS LONGITUDE")
speed_column = find_column("GPS SPEED")
accel_x_column = find_column("ACCELEROMETER X")
accel_y_column = find_column("ACCELEROMETER Y")
accel_z_column = find_column("ACCELEROMETER Z")

time = pd.to_numeric(df[time_column], errors="coerce") / 1000

latitude = pd.to_numeric(df[latitude_column], errors="coerce")
longitude = pd.to_numeric(df[longitude_column], errors="coerce")

accel_x = pd.to_numeric(df[accel_x_column], errors="coerce")
accel_y = pd.to_numeric(df[accel_y_column], errors="coerce")
accel_z = pd.to_numeric(df[accel_z_column], errors="coerce")

start_time = 3562
end_time = 3622

section = df[
    (time >= start_time) &
    (time <= end_time)
].copy()

section_time = time[
    (time >= start_time) &
    (time <= end_time)
].to_numpy()

section_lat = latitude[
    (time >= start_time) &
    (time <= end_time)
].to_numpy()

section_lon = longitude[
    (time >= start_time) &
    (time <= end_time)
].to_numpy()

section_ax = accel_x[
    (time >= start_time) &
    (time <= end_time)
].to_numpy()

section_ay = accel_y[
    (time >= start_time) &
    (time <= end_time)
].to_numpy()

section_az = accel_z[
    (time >= start_time) &
    (time <= end_time)
].to_numpy()

section_time = section_time - section_time[0]

lat0 = section_lat[0]
lon0 = section_lon[0]

earth_radius = 6371000

north = np.radians(section_lat - lat0) * earth_radius
east = (
    np.radians(section_lon - lon0)
    * earth_radius
    * np.cos(np.radians(lat0))
)

dt = np.diff(section_time, prepend=section_time[0])

accel_forward = section_ax
accel_side = section_ay

accel_forward = accel_forward - np.mean(accel_forward[:20])
accel_side = accel_side - np.mean(accel_side[:20])

velocity_forward = np.zeros(len(accel_forward))
velocity_side = np.zeros(len(accel_side))

position_forward = np.zeros(len(accel_forward))
position_side = np.zeros(len(accel_side))

for i in range(1, len(accel_forward)):

    velocity_forward[i] = (
        velocity_forward[i - 1]
        + accel_forward[i] * dt[i]
    )

    velocity_side[i] = (
        velocity_side[i - 1]
        + accel_side[i] * dt[i]
    )

    position_forward[i] = (
        position_forward[i - 1]
        + velocity_forward[i] * dt[i]
    )

    position_side[i] = (
        position_side[i - 1]
        + velocity_side[i] * dt[i]
    )

print("======================================")
print("       DEAD RECKONING BASELINE")
print("======================================")
print()

print("Blackout interval:")
print(f"{start_time}s → {end_time}s")

print()

print("Number of samples:", len(section_time))

print()

print("Initial GPS position:")
print("Latitude :", lat0)
print("Longitude:", lon0)

print()

final_gps_distance = np.sqrt(
    north[-1] ** 2 +
    east[-1] ** 2
)

final_dr_distance = np.sqrt(
    position_forward[-1] ** 2 +
    position_side[-1] ** 2
)

error = np.sqrt(
    (position_forward[-1] - north[-1]) ** 2 +
    (position_side[-1] - east[-1]) ** 2
)

print("Actual GPS displacement:")
print(f"{final_gps_distance:.2f} meters")

print()

print("Dead reckoning displacement:")
print(f"{final_dr_distance:.2f} meters")

print()

print("Final position error:")
print(f"{error:.2f} meters")

print()

if final_gps_distance > 0:
    drift_percentage = (
        error / final_gps_distance
    ) * 100

    print("Drift:")
    print(f"{drift_percentage:.2f}%")

print()

print("======================================")

plt.figure(figsize=(10, 6))

plt.plot(
    east,
    north,
    label="Actual GPS"
)

plt.plot(
    position_side,
    position_forward,
    label="Baseline Dead Reckoning"
)

plt.xlabel("East displacement (m)")
plt.ylabel("North displacement (m)")
plt.title("GNSS-Denied Navigation Baseline")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print()
print("Baseline experiment completed.")