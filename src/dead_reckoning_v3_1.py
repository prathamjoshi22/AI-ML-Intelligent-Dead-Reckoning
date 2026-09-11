import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.ndimage import uniform_filter1d


# ============================================================
# CONFIGURATION
# ============================================================

START_TIME = 3562.0
DURATION = 60.0

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_csv(filename):
    matches = list(DATA_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} inside {DATA_DIR}"
        )

    return matches[0]


def find_column(df, text):
    text = text.lower()

    for col in df.columns:
        if text in col.lower():
            return col

    raise KeyError(
        f"Could not find column containing: {text}"
    )


def latlon_to_local(lat, lon, lat0, lon0):
    R = 6371000.0

    north = np.deg2rad(lat - lat0) * R
    east = (
        np.deg2rad(lon - lon0)
        * R
        * np.cos(np.deg2rad(lat0))
    )

    return east, north


# ============================================================
# LOAD DATA
# ============================================================

smart_path = find_csv("S-S1.csv")
vehicle_path = find_csv("V-S1.csv")

print("=" * 60)
print("GNSS-DENIED DEAD RECKONING BASELINE V3.1")
print("CONTROLLED INITIALIZATION TEST")
print("=" * 60)

print("\nSmartphone file:")
print(smart_path)

print("\nVehicle file:")
print(vehicle_path)

smart = pd.read_csv(smart_path, encoding="cp1252")
vehicle = pd.read_csv(vehicle_path, encoding="cp1252")

print("\nSmartphone shape:", smart.shape)
print("Vehicle shape:", vehicle.shape)


# ============================================================
# FIND IMPORTANT COLUMNS
# ============================================================

smart_time_col = find_column(
    smart,
    "TIME SINCE START"
)

acc_x_col = find_column(
    smart,
    "ACCELEROMETER X"
)

acc_y_col = find_column(
    smart,
    "ACCELEROMETER Y"
)

acc_z_col = find_column(
    smart,
    "ACCELEROMETER Z"
)

gravity_x_col = find_column(
    smart,
    "GRAVITY X"
)

gravity_y_col = find_column(
    smart,
    "GRAVITY Y"
)

gravity_z_col = find_column(
    smart,
    "GRAVITY Z"
)

yaw_col = find_column(
    smart,
    "ORIENTATION (Yaw)"
)

pitch_col = find_column(
    smart,
    "ORIENTATION (Pitch)"
)

roll_col = find_column(
    smart,
    "ORIENTATION (Roll"
)

gps_lat_col = find_column(
    smart,
    "GPS LATITUDE"
)

gps_lon_col = find_column(
    smart,
    "GPS LONGITUDE"
)

vehicle_time_col = find_column(
    vehicle,
    "Time"
)

vehicle_lat_col = find_column(
    vehicle,
    "Latitude"
)

vehicle_lon_col = find_column(
    vehicle,
    "Longitude"
)

vehicle_speed_col = find_column(
    vehicle,
    "Velocity"
)

vehicle_heading_col = find_column(
    vehicle,
    "Heading"
)


# ============================================================
# CONVERT TIME
# ============================================================

smart_time = pd.to_numeric(
    smart[smart_time_col],
    errors="coerce"
) / 1000.0

vehicle_time = pd.to_numeric(
    vehicle[vehicle_time_col],
    errors="coerce"
)


# ============================================================
# SELECT SAME OUTAGE WINDOW
# ============================================================

end_time = START_TIME + DURATION

smart_mask = (
    (smart_time >= START_TIME) &
    (smart_time <= end_time)
)

smart_segment = smart.loc[smart_mask].copy()

smart_segment["relative_time"] = (
    smart_time.loc[smart_mask].values - START_TIME
)

print("\nSelected smartphone samples:")
print(len(smart_segment))

if len(smart_segment) < 10:
    raise RuntimeError(
        "Not enough smartphone samples in selected interval."
    )


# ============================================================
# MAP SMARTPHONE ROWS TO VEHICLE ROWS
# ============================================================

start_index = smart_segment.index[0]
end_index = smart_segment.index[-1]

vehicle_segment = vehicle.iloc[
    start_index:end_index + 1
].copy()

print("Vehicle samples:", len(vehicle_segment))

if len(vehicle_segment) != len(smart_segment):
    raise RuntimeError(
        "Smartphone and vehicle segment lengths do not match."
    )


# ============================================================
# TIME
# ============================================================

t = smart_segment["relative_time"].to_numpy()

dt = np.diff(t, prepend=t[0])

dt[0] = np.median(
    np.diff(t)
)

print("\nAverage dt:", np.mean(dt))
print("Minimum dt:", np.min(dt))
print("Maximum dt:", np.max(dt))


# ============================================================
# CONTROLLED INITIALIZATION
# ============================================================

initial_vehicle_speed_kmh = float(
    pd.to_numeric(
        vehicle_segment[vehicle_speed_col],
        errors="coerce"
    ).iloc[0]
)

initial_vehicle_heading = float(
    pd.to_numeric(
        vehicle_segment[vehicle_heading_col],
        errors="coerce"
    ).iloc[0]
)

initial_lat = float(
    pd.to_numeric(
        vehicle_segment[vehicle_lat_col],
        errors="coerce"
    ).iloc[0]
)

initial_lon = float(
    pd.to_numeric(
        vehicle_segment[vehicle_lon_col],
        errors="coerce"
    ).iloc[0]
)

initial_speed = (
    initial_vehicle_speed_kmh / 3.6
)

heading_rad = np.deg2rad(
    initial_vehicle_heading
)


# ============================================================
# INITIAL VELOCITY
# Heading convention:
# 0 degrees = North
# 90 degrees = East
# ============================================================

initial_velocity_east = (
    initial_speed * np.sin(heading_rad)
)

initial_velocity_north = (
    initial_speed * np.cos(heading_rad)
)

velocity = np.array([
    initial_velocity_east,
    initial_velocity_north
])

position = np.array([
    0.0,
    0.0
])


# ============================================================
# SMARTPHONE SENSOR DATA
# ============================================================

acc = smart_segment[
    [
        acc_x_col,
        acc_y_col,
        acc_z_col
    ]
].to_numpy(dtype=float)

gravity = smart_segment[
    [
        gravity_x_col,
        gravity_y_col,
        gravity_z_col
    ]
].to_numpy(dtype=float)

yaw = pd.to_numeric(
    smart_segment[yaw_col],
    errors="coerce"
).to_numpy(dtype=float)

pitch = pd.to_numeric(
    smart_segment[pitch_col],
    errors="coerce"
).to_numpy(dtype=float)

roll = pd.to_numeric(
    smart_segment[roll_col],
    errors="coerce"
).to_numpy(dtype=float)


# ============================================================
# GRAVITY COMPENSATION
# ============================================================

linear_acc = acc - gravity


# ============================================================
# FILTER SENSOR NOISE
# ============================================================

filtered_acc = np.zeros_like(linear_acc)

for axis in range(3):
    filtered_acc[:, axis] = uniform_filter1d(
        linear_acc[:, axis],
        size=5,
        mode="nearest"
    )


# ============================================================
# DEAD RECKONING
# ============================================================

positions = []

positions.append(
    position.copy()
)

for i in range(1, len(filtered_acc)):

    # --------------------------------------------------------
    # Device frame -> world frame
    #
    # ZYX:
    # yaw -> pitch -> roll
    # --------------------------------------------------------

    rotation = Rotation.from_euler(
        "ZYX",
        [
            yaw[i],
            pitch[i],
            roll[i]
        ],
        degrees=True
    )

    world_acc = rotation.apply(
        filtered_acc[i]
    )

    # Android/world convention used here:
    # X = East
    # Y = North
    # Z = Up
    #
    # We only use horizontal acceleration.

    east_acc = world_acc[0]
    north_acc = world_acc[1]

    acceleration_horizontal = np.array([
        east_acc,
        north_acc
    ])

    # --------------------------------------------------------
    # Integrate acceleration -> velocity
    # --------------------------------------------------------

    velocity += (
        acceleration_horizontal * dt[i]
    )

    # --------------------------------------------------------
    # Integrate velocity -> position
    # --------------------------------------------------------

    position += (
        velocity * dt[i]
    )

    positions.append(
        position.copy()
    )


positions = np.array(positions)


# ============================================================
# VEHICLE GROUND TRUTH
# ============================================================

vehicle_lat = pd.to_numeric(
    vehicle_segment[vehicle_lat_col],
    errors="coerce"
).to_numpy()

vehicle_lon = pd.to_numeric(
    vehicle_segment[vehicle_lon_col],
    errors="coerce"
).to_numpy()

truth_east, truth_north = latlon_to_local(
    vehicle_lat,
    vehicle_lon,
    vehicle_lat[0],
    vehicle_lon[0]
)


# ============================================================
# GROUND TRUTH DISTANCES
# ============================================================

truth_endpoint_distance = np.sqrt(
    truth_east[-1] ** 2 +
    truth_north[-1] ** 2
)

truth_travel_distance = np.sum(
    np.sqrt(
        np.diff(truth_east) ** 2 +
        np.diff(truth_north) ** 2
    )
)


# ============================================================
# DR RESULTS
# ============================================================

dr_final_east = positions[-1, 0]
dr_final_north = positions[-1, 1]

dr_endpoint_distance = np.sqrt(
    dr_final_east ** 2 +
    dr_final_north ** 2
)

final_error = np.sqrt(
    (dr_final_east - truth_east[-1]) ** 2 +
    (dr_final_north - truth_north[-1]) ** 2
)

drift_percentage = (
    final_error /
    truth_travel_distance
) * 100


# ============================================================
# FINAL SPEED
# ============================================================

final_speed = np.sqrt(
    velocity[0] ** 2 +
    velocity[1] ** 2
)

final_speed_kmh = final_speed * 3.6


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("CONTROLLED INITIALIZATION")
print("=" * 60)

print(
    f"Initial vehicle speed   : "
    f"{initial_vehicle_speed_kmh:.3f} km/h"
)

print(
    f"Initial vehicle heading : "
    f"{initial_vehicle_heading:.3f} degrees"
)

print(
    f"Initial latitude        : "
    f"{initial_lat:.8f}"
)

print(
    f"Initial longitude       : "
    f"{initial_lon:.8f}"
)

print("\n")
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"Ground truth endpoint displacement : "
    f"{truth_endpoint_distance:.2f} m"
)

print(
    f"Ground truth traveled distance     : "
    f"{truth_travel_distance:.2f} m"
)

print(
    f"Dead reckoning displacement        : "
    f"{dr_endpoint_distance:.2f} m"
)

print(
    f"Final position error               : "
    f"{final_error:.2f} m"
)

print(
    f"Drift relative to traveled path   : "
    f"{drift_percentage:.2f}%"
)

print(
    f"Final DR speed                     : "
    f"{final_speed_kmh:.2f} km/h"
)


# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame({
    "time": t,
    "truth_east": truth_east,
    "truth_north": truth_north,
    "dr_east": positions[:, 0],
    "dr_north": positions[:, 1]
})

result_path = BASE_DIR / "v3_1_results.csv"

result_df.to_csv(
    result_path,
    index=False
)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12, 8))

plt.plot(
    truth_east,
    truth_north,
    label="Vehicle Ground Truth",
    linewidth=2
)

plt.plot(
    positions[:, 0],
    positions[:, 1],
    label="IMU Dead Reckoning V3.1",
    linewidth=2
)

plt.scatter(
    0,
    0,
    s=100,
    label="Start"
)

plt.xlabel("East (meters)")
plt.ylabel("North (meters)")

plt.title(
    "GNSS-Denied Dead Reckoning Baseline V3.1\n"
    "Controlled Initialization Test"
)

plt.grid(True)
plt.axis("equal")
plt.legend()

plt.tight_layout()

plot_path = BASE_DIR / "v3_1_controlled.png"

plt.savefig(
    plot_path,
    dpi=200
)

plt.show()


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 60)
print("FILES SAVED")
print("=" * 60)

print(
    f"Results CSV : {result_path}"
)

print(
    f"Graph       : {plot_path}"
)

print("\nV3.1 CONTROLLED TEST COMPLETE.")