import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "Data"

smartphone_files = list(data_folder.rglob("S-S1.csv"))
vehicle_files = list(data_folder.rglob("V-S1.csv"))

if not smartphone_files:
    raise FileNotFoundError("S-S1.csv not found.")

if not vehicle_files:
    raise FileNotFoundError("V-S1.csv not found.")

smartphone = pd.read_csv(
    smartphone_files[0],
    encoding="cp1252"
)

vehicle = pd.read_csv(
    vehicle_files[0],
    encoding="cp1252"
)

smartphone.columns = smartphone.columns.str.strip()
vehicle.columns = vehicle.columns.str.strip()

# ------------------------------------------------
# SELECT GNSS OUTAGE
# ------------------------------------------------

start_index = 35591
end_index = 36191

s = smartphone.iloc[start_index:end_index + 1].copy()
v = vehicle.iloc[start_index:end_index + 1].copy()

print("==============================================")
print("       IMU DEAD RECKONING BASELINE V2")
print("==============================================")
print()

print("Samples:", len(s))
print("Duration:",
      s["TIME SINCE START (ms)"].iloc[-1] / 1000 -
      s["TIME SINCE START (ms)"].iloc[0] / 1000,
      "seconds")
print()

# ------------------------------------------------
# TIME
# ------------------------------------------------

time = (
    s["TIME SINCE START (ms)"].values / 1000
)

dt = np.diff(time)

dt = np.append(dt, np.mean(dt))

# ------------------------------------------------
# SMARTPHONE ACCELEROMETER
# ------------------------------------------------

ax = s["ACCELEROMETER X (m/s²)"].values
ay = s["ACCELEROMETER Y (m/s²)"].values
az = s["ACCELEROMETER Z (m/s²)"].values

# ------------------------------------------------
# GRAVITY
# ------------------------------------------------

gx = s["GRAVITY X (m/s²)"].values
gy = s["GRAVITY Y (m/s²)"].values
gz = s["GRAVITY Z (m/s²)"].values

# ------------------------------------------------
# LINEAR ACCELERATION
# ------------------------------------------------

linear_x = ax - gx
linear_y = ay - gy
linear_z = az - gz

# ------------------------------------------------
# REMOVE SENSOR BIAS
# ------------------------------------------------

bias_samples = min(100, len(s))

bias_x = np.mean(linear_x[:bias_samples])
bias_y = np.mean(linear_y[:bias_samples])
bias_z = np.mean(linear_z[:bias_samples])

linear_x = linear_x - bias_x
linear_y = linear_y - bias_y
linear_z = linear_z - bias_z

# ------------------------------------------------
# SMOOTHING
# ------------------------------------------------

window = 5

linear_x = (
    pd.Series(linear_x)
    .rolling(window, center=True)
    .mean()
    .bfill()
    .ffill()
    .values
)

linear_y = (
    pd.Series(linear_y)
    .rolling(window, center=True)
    .mean()
    .bfill()
    .ffill()
    .values
)

linear_z = (
    pd.Series(linear_z)
    .rolling(window, center=True)
    .mean()
    .bfill()
    .ffill()
    .values
)

# ------------------------------------------------
# VEHICLE HEADING
#
# IMPORTANT:
# Used ONLY to establish initial coordinate
# alignment. It is NOT used during the outage.
# ------------------------------------------------

vehicle_heading = v["Heading (degrees)"].values

initial_heading = vehicle_heading[0]

heading_rad = np.radians(initial_heading)

# ------------------------------------------------
# INITIAL ROTATION
#
# Rotate smartphone horizontal acceleration
# into approximate vehicle/world coordinates.
# ------------------------------------------------

forward = linear_x
sideways = linear_y

north_acc = (
    forward * np.cos(heading_rad)
    - sideways * np.sin(heading_rad)
)

east_acc = (
    forward * np.sin(heading_rad)
    + sideways * np.cos(heading_rad)
)

# ------------------------------------------------
# DEAD RECKONING INTEGRATION
# ------------------------------------------------

velocity_north = np.zeros(len(s))
velocity_east = np.zeros(len(s))

position_north = np.zeros(len(s))
position_east = np.zeros(len(s))

for i in range(1, len(s)):

    velocity_north[i] = (
        velocity_north[i - 1]
        + north_acc[i] * dt[i]
    )

    velocity_east[i] = (
        velocity_east[i - 1]
        + east_acc[i] * dt[i]
    )

    position_north[i] = (
        position_north[i - 1]
        + velocity_north[i] * dt[i]
    )

    position_east[i] = (
        position_east[i - 1]
        + velocity_east[i] * dt[i]
    )

# ------------------------------------------------
# VEHICLE GROUND TRUTH
# ------------------------------------------------

lat = v["Latitude (degrees)"].values
lon = v["Longitude (degrees)"].values

lat0 = np.radians(lat[0])

truth_north = (
    lat - lat[0]
) * 111320

truth_east = (
    lon - lon[0]
) * 111320 * np.cos(lat0)

# ------------------------------------------------
# FINAL ERROR
# ------------------------------------------------

error = np.sqrt(
    (position_north - truth_north) ** 2
    +
    (position_east - truth_east) ** 2
)

final_error = error[-1]

truth_distance = np.sqrt(
    truth_north[-1] ** 2
    +
    truth_east[-1] ** 2
)

if truth_distance > 0:
    drift = (
        final_error /
        truth_distance *
        100
    )
else:
    drift = 0

# ------------------------------------------------
# RESULTS
# ------------------------------------------------

print("==============================================")
print("                 RESULTS")
print("==============================================")
print()

print("Initial vehicle heading:",
      initial_heading,
      "degrees")

print()

print("Ground truth displacement:",
      round(truth_distance, 2),
      "meters")

print()

print("Dead reckoning displacement:",
      round(
          np.sqrt(
              position_north[-1] ** 2 +
              position_east[-1] ** 2
          ),
          2
      ),
      "meters")

print()

print("Final position error:",
      round(final_error, 2),
      "meters")

print()

print("Drift:",
      round(drift, 2),
      "%")

print()

# ------------------------------------------------
# PLOT
# ------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    truth_east,
    truth_north,
    label="Vehicle Ground Truth"
)

plt.plot(
    position_east,
    position_north,
    label="IMU Dead Reckoning"
)

plt.xlabel("East (meters)")
plt.ylabel("North (meters)")
plt.title("GNSS-Denied Dead Reckoning Baseline")

plt.legend()
plt.grid()

plt.axis("equal")

plt.tight_layout()

plt.show()