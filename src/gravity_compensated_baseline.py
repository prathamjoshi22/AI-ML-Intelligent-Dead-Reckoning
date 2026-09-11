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

accel_x_column = find_column("ACCELEROMETER X")
accel_y_column = find_column("ACCELEROMETER Y")
accel_z_column = find_column("ACCELEROMETER Z")

gravity_x_column = find_column("GRAVITY X")
gravity_y_column = find_column("GRAVITY Y")
gravity_z_column = find_column("GRAVITY Z")

time = pd.to_numeric(
    df[time_column],
    errors="coerce"
) / 1000

latitude = pd.to_numeric(
    df[latitude_column],
    errors="coerce"
)

longitude = pd.to_numeric(
    df[longitude_column],
    errors="coerce"
)

ax = pd.to_numeric(
    df[accel_x_column],
    errors="coerce"
)

ay = pd.to_numeric(
    df[accel_y_column],
    errors="coerce"
)

az = pd.to_numeric(
    df[accel_z_column],
    errors="coerce"
)

gx = pd.to_numeric(
    df[gravity_x_column],
    errors="coerce"
)

gy = pd.to_numeric(
    df[gravity_y_column],
    errors="coerce"
)

gz = pd.to_numeric(
    df[gravity_z_column],
    errors="coerce"
)

start_time = 3562
end_time = 3622

mask = (
    (time >= start_time) &
    (time <= end_time)
)

section_time = time[mask].to_numpy()

section_lat = latitude[mask].to_numpy()
section_lon = longitude[mask].to_numpy()

section_ax = ax[mask].to_numpy()
section_ay = ay[mask].to_numpy()
section_az = az[mask].to_numpy()

section_gx = gx[mask].to_numpy()
section_gy = gy[mask].to_numpy()
section_gz = gz[mask].to_numpy()

section_time = section_time - section_time[0]

lat0 = section_lat[0]
lon0 = section_lon[0]

earth_radius = 6371000

north = (
    np.radians(section_lat - lat0)
    * earth_radius
)

east = (
    np.radians(section_lon - lon0)
    * earth_radius
    * np.cos(np.radians(lat0))
)

linear_x = section_ax - section_gx
linear_y = section_ay - section_gy
linear_z = section_az - section_gz

dt = np.diff(
    section_time,
    prepend=section_time[0]
)

velocity_x = np.zeros(len(linear_x))
velocity_y = np.zeros(len(linear_y))

position_x = np.zeros(len(linear_x))
position_y = np.zeros(len(linear_y))

for i in range(1, len(linear_x)):

    velocity_x[i] = (
        velocity_x[i - 1]
        + linear_x[i] * dt[i]
    )

    velocity_y[i] = (
        velocity_y[i - 1]
        + linear_y[i] * dt[i]
    )

    position_x[i] = (
        position_x[i - 1]
        + velocity_x[i] * dt[i]
    )

    position_y[i] = (
        position_y[i - 1]
        + velocity_y[i] * dt[i]
    )

gps_displacement = np.sqrt(
    north[-1] ** 2 +
    east[-1] ** 2
)

dr_displacement = np.sqrt(
    position_x[-1] ** 2 +
    position_y[-1] ** 2
)

position_error = np.sqrt(
    (position_x[-1] - north[-1]) ** 2 +
    (position_y[-1] - east[-1]) ** 2
)

drift_percentage = (
    position_error /
    gps_displacement
) * 100

print("======================================")
print("   GRAVITY COMPENSATED BASELINE")
print("======================================")
print()

print("GNSS blackout:")
print(f"{start_time}s → {end_time}s")

print()

print("Samples:", len(section_time))

print()

print("Gravity magnitude:")

gravity_magnitude = np.sqrt(
    section_gx ** 2 +
    section_gy ** 2 +
    section_gz ** 2
)

print(
    "Average:",
    gravity_magnitude.mean()
)

print()

print("Actual GPS displacement:")
print(f"{gps_displacement:.2f} meters")

print()

print("Dead reckoning displacement:")
print(f"{dr_displacement:.2f} meters")

print()

print("Final position error:")
print(f"{position_error:.2f} meters")

print()

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
    position_y,
    position_x,
    label="Gravity Compensated DR"
)

plt.xlabel("East displacement (m)")
plt.ylabel("North displacement (m)")

plt.title(
    "GNSS-Denied Navigation "
    "with Gravity Compensation"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

print()
print("Gravity compensated baseline completed.")