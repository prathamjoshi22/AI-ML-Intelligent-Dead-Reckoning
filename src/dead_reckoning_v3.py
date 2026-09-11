import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation


# ============================================================
# CONFIGURATION
# ============================================================

OUTAGE_START = 3562.0
OUTAGE_END = 3622.0

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "Data"


# ============================================================
# FIND DATA FILES
# ============================================================

smartphone_files = list(DATA_FOLDER.rglob("S-S1.csv"))
vehicle_files = list(DATA_FOLDER.rglob("V-S1.csv"))

if not smartphone_files:
    raise FileNotFoundError("S-S1.csv not found")

if not vehicle_files:
    raise FileNotFoundError("V-S1.csv not found")

smartphone_path = smartphone_files[0]
vehicle_path = vehicle_files[0]

print("Smartphone:", smartphone_path)
print("Vehicle:", vehicle_path)


# ============================================================
# LOAD DATA
# ============================================================

smart = pd.read_csv(
    smartphone_path,
    encoding="cp1252"
)

vehicle = pd.read_csv(
    vehicle_path,
    encoding="cp1252"
)

smart.columns = smart.columns.str.strip()
vehicle.columns = vehicle.columns.str.strip()


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(df, text):

    matches = [
        c for c in df.columns
        if text.lower() in c.lower()
    ]

    if not matches:
        raise KeyError(
            f"Could not find column containing '{text}'\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return matches[0]


# ============================================================
# SMARTPHONE COLUMNS
# ============================================================

time_col = find_column(
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

grav_x_col = find_column(
    smart,
    "GRAVITY X"
)

grav_y_col = find_column(
    smart,
    "GRAVITY Y"
)

grav_z_col = find_column(
    smart,
    "GRAVITY Z"
)

gyro_yaw_col = find_column(
    smart,
    "GYROSCOPE Yaw"
)

gyro_pitch_col = find_column(
    smart,
    "GYROSCOPE Pitch"
)

gyro_roll_col = find_column(
    smart,
    "GYROSCOPE Roll"
)

orient_yaw_col = find_column(
    smart,
    "ORIENTATION (Yaw)"
)

orient_pitch_col = find_column(
    smart,
    "ORIENTATION (Pitch)"
)

orient_roll_col = find_column(
    smart,
    "ORIENTATION (Roll"
)

gps_speed_col = find_column(
    smart,
    "GPS SPEED"
)

gps_heading_col = find_column(
    smart,
    "GPS ORIENTATION"
)

gps_lat_col = find_column(
    smart,
    "GPS LATITUDE"
)

gps_lon_col = find_column(
    smart,
    "GPS LONGITUDE"
)


# ============================================================
# VEHICLE COLUMNS
# ============================================================

vehicle_lat_col = find_column(
    vehicle,
    "Latitude"
)

vehicle_lon_col = find_column(
    vehicle,
    "Longitude"
)

vehicle_heading_col = find_column(
    vehicle,
    "Heading"
)

vehicle_speed_col = find_column(
    vehicle,
    "Velocity"
)


# ============================================================
# SMARTPHONE TIME
# ============================================================

smart_time = pd.to_numeric(
    smart[time_col],
    errors="coerce"
)

# Convert milliseconds to seconds
if smart_time.max() > 100000:

    smart_time = smart_time / 1000.0


# ============================================================
# FIND GNSS OUTAGE WINDOW
# ============================================================

start_idx = int(
    np.abs(
        smart_time - OUTAGE_START
    ).argmin()
)

end_idx = int(
    np.abs(
        smart_time - OUTAGE_END
    ).argmin()
)


print()
print("==============================================")
print("              V3 OUTAGE WINDOW")
print("==============================================")

print(
    "Start index:",
    start_idx
)

print(
    "End index:",
    end_idx
)

print(
    "Start time:",
    smart_time.iloc[start_idx]
)

print(
    "End time:",
    smart_time.iloc[end_idx]
)


# ============================================================
# EXTRACT OUTAGE DATA
# ============================================================

s = smart.iloc[
    start_idx:end_idx + 1
].copy()

v = vehicle.iloc[
    start_idx:end_idx + 1
].copy()

t = smart_time.iloc[
    start_idx:end_idx + 1
].to_numpy()


# Calculate time differences

dt = np.diff(t)

dt = np.clip(
    dt,
    0.05,
    0.2
)


print()
print(
    "Samples:",
    len(s)
)

print(
    "Average dt:",
    np.mean(dt)
)


# ============================================================
# INITIAL STATE
# ============================================================

# Last sample before GNSS outage

pre_idx = max(
    0,
    start_idx - 1
)


initial_speed_kmh = float(
    pd.to_numeric(
        smart[gps_speed_col],
        errors="coerce"
    ).iloc[pre_idx]
)


initial_heading_deg = float(
    pd.to_numeric(
        smart[gps_heading_col],
        errors="coerce"
    ).iloc[pre_idx]
)


initial_lat = float(
    pd.to_numeric(
        smart[gps_lat_col],
        errors="coerce"
    ).iloc[start_idx]
)


initial_lon = float(
    pd.to_numeric(
        smart[gps_lon_col],
        errors="coerce"
    ).iloc[start_idx]
)


vehicle_initial_speed = float(
    pd.to_numeric(
        vehicle[vehicle_speed_col],
        errors="coerce"
    ).iloc[start_idx]
)


vehicle_initial_heading = float(
    pd.to_numeric(
        vehicle[vehicle_heading_col],
        errors="coerce"
    ).iloc[start_idx]
)


print()
print("==============================================")
print("             INITIAL STATE")
print("==============================================")

print(
    f"Initial smartphone GPS speed: "
    f"{initial_speed_kmh:.3f} km/h"
)

print(
    f"Vehicle reference speed: "
    f"{vehicle_initial_speed:.3f} km/h"
)

print(
    f"Initial smartphone heading: "
    f"{initial_heading_deg:.3f} degrees"
)

print(
    f"Vehicle reference heading: "
    f"{vehicle_initial_heading:.3f} degrees"
)

print(
    f"Initial latitude: "
    f"{initial_lat}"
)

print(
    f"Initial longitude: "
    f"{initial_lon}"
)

print(
    f"Speed difference: "
    f"{abs(initial_speed_kmh - vehicle_initial_speed):.3f} km/h"
)


# ============================================================
# INITIAL PHONE ORIENTATION
# ============================================================

yaw0 = float(
    pd.to_numeric(
        s[orient_yaw_col],
        errors="coerce"
    ).iloc[0]
)

pitch0 = float(
    pd.to_numeric(
        s[orient_pitch_col],
        errors="coerce"
    ).iloc[0]
)

roll0 = float(
    pd.to_numeric(
        s[orient_roll_col],
        errors="coerce"
    ).iloc[0]
)


print()
print("==============================================")
print("             PHONE ORIENTATION")
print("==============================================")

print(
    f"Yaw:   {yaw0:.3f} degrees"
)

print(
    f"Pitch: {pitch0:.3f} degrees"
)

print(
    f"Roll:  {roll0:.3f} degrees"
)


# ============================================================
# INITIAL VELOCITY
# ============================================================

speed_ms = (
    initial_speed_kmh / 3.6
)

heading_rad = np.radians(
    initial_heading_deg
)


# North / East velocity

velocity_north = (
    speed_ms *
    np.cos(heading_rad)
)

velocity_east = (
    speed_ms *
    np.sin(heading_rad)
)


# ============================================================
# ACCELEROMETER DATA
# ============================================================

acc = np.column_stack(
    [
        pd.to_numeric(
            s[acc_x_col],
            errors="coerce"
        ).to_numpy(),

        pd.to_numeric(
            s[acc_y_col],
            errors="coerce"
        ).to_numpy(),

        pd.to_numeric(
            s[acc_z_col],
            errors="coerce"
        ).to_numpy()
    ]
)


# ============================================================
# GRAVITY DATA
# ============================================================

gravity = np.column_stack(
    [
        pd.to_numeric(
            s[grav_x_col],
            errors="coerce"
        ).to_numpy(),

        pd.to_numeric(
            s[grav_y_col],
            errors="coerce"
        ).to_numpy(),

        pd.to_numeric(
            s[grav_z_col],
            errors="coerce"
        ).to_numpy()
    ]
)


# ============================================================
# REMOVE GRAVITY
# ============================================================

linear_acc = (
    acc - gravity
)


# ============================================================
# ORIENTATION
# ============================================================

orientation_yaw = pd.to_numeric(
    s[orient_yaw_col],
    errors="coerce"
).to_numpy()

orientation_pitch = pd.to_numeric(
    s[orient_pitch_col],
    errors="coerce"
).to_numpy()

orientation_roll = pd.to_numeric(
    s[orient_roll_col],
    errors="coerce"
).to_numpy()


# ============================================================
# GYROSCOPE
# ============================================================

gyro_yaw = pd.to_numeric(
    s[gyro_yaw_col],
    errors="coerce"
).to_numpy()

gyro_pitch = pd.to_numeric(
    s[gyro_pitch_col],
    errors="coerce"
).to_numpy()

gyro_roll = pd.to_numeric(
    s[gyro_roll_col],
    errors="coerce"
).to_numpy()


# ============================================================
# ROTATION FUNCTION
# ============================================================

def phone_to_world(
    yaw,
    pitch,
    roll
):

    rotation = Rotation.from_euler(
        "ZYX",
        [
            yaw,
            pitch,
            roll
        ],
        degrees=True
    )

    return rotation


# ============================================================
# INITIAL HEADING
# ============================================================

current_heading = (
    initial_heading_deg
)


# ============================================================
# TRAJECTORY ARRAYS
# ============================================================

north = np.zeros(
    len(s)
)

east = np.zeros(
    len(s)
)


# ============================================================
# INITIAL FILTER VALUES
# ============================================================

previous_acc_north = 0.0
previous_acc_east = 0.0


# ============================================================
# MAIN DEAD-RECKONING LOOP
# ============================================================

for i in range(
    1,
    len(s)
):

    dti = dt[i - 1]


    # --------------------------------------------------------
    # PHONE ORIENTATION
    # --------------------------------------------------------

    yaw = orientation_yaw[i]

    pitch = orientation_pitch[i]

    roll = orientation_roll[i]


    # --------------------------------------------------------
    # PHONE → WORLD ROTATION
    # --------------------------------------------------------

    rotation = phone_to_world(
        yaw,
        pitch,
        roll
    )


    world_acc = rotation.apply(
        linear_acc[i]
    )


    # --------------------------------------------------------
    # GYROSCOPE HEADING UPDATE
    # --------------------------------------------------------

    yaw_rate = gyro_yaw[i]

    current_heading += (
        np.degrees(
            yaw_rate * dti
        )
    )


    # Keep heading between -180 and +180

    current_heading = (
        current_heading + 180
    ) % 360 - 180


    # --------------------------------------------------------
    # WORLD ACCELERATION
    # --------------------------------------------------------

    acc_north = world_acc[1]

    acc_east = world_acc[0]


    # --------------------------------------------------------
    # LOW-PASS FILTER
    # --------------------------------------------------------

    acc_north = (
        0.8 * previous_acc_north
        +
        0.2 * acc_north
    )

    acc_east = (
        0.8 * previous_acc_east
        +
        0.2 * acc_east
    )


    previous_acc_north = (
        acc_north
    )

    previous_acc_east = (
        acc_east
    )


    # --------------------------------------------------------
    # VELOCITY UPDATE
    # --------------------------------------------------------

    velocity_north += (
        acc_north * dti
    )

    velocity_east += (
        acc_east * dti
    )


    # --------------------------------------------------------
    # POSITION UPDATE
    # --------------------------------------------------------

    north[i] = (
        north[i - 1]
        +
        velocity_north * dti
    )

    east[i] = (
        east[i - 1]
        +
        velocity_east * dti
    )


# ============================================================
# VEHICLE GROUND TRUTH
# ============================================================

vehicle_lat = pd.to_numeric(
    v[vehicle_lat_col],
    errors="coerce"
).to_numpy()

vehicle_lon = pd.to_numeric(
    v[vehicle_lon_col],
    errors="coerce"
).to_numpy()


lat0 = np.radians(
    vehicle_lat[0]
)


truth_north = (
    vehicle_lat -
    vehicle_lat[0]
) * 111320.0


truth_east = (
    vehicle_lon -
    vehicle_lon[0]
) * 111320.0 * np.cos(
    lat0
)


# ============================================================
# DISPLACEMENT
# ============================================================

truth_displacement = np.sqrt(
    truth_north[-1] ** 2
    +
    truth_east[-1] ** 2
)


dr_displacement = np.sqrt(
    north[-1] ** 2
    +
    east[-1] ** 2
)


# ============================================================
# FINAL ERROR
# ============================================================

final_error = np.sqrt(
    (
        north[-1]
        -
        truth_north[-1]
    ) ** 2
    +
    (
        east[-1]
        -
        truth_east[-1]
    ) ** 2
)


# ============================================================
# DRIFT
# ============================================================

drift_percentage = (
    final_error /
    truth_displacement
) * 100


# ============================================================
# RESULTS
# ============================================================

print()
print("==============================================")
print("       IMU DEAD RECKONING BASELINE V3")
print("==============================================")

print()
print(
    f"Samples: {len(s)}"
)

print(
    f"Duration: {t[-1] - t[0]:.3f} seconds"
)

print()
print("==============================================")
print("                 RESULTS")
print("==============================================")

print(
    f"Initial speed: "
    f"{initial_speed_kmh:.2f} km/h"
)

print(
    f"Initial heading: "
    f"{initial_heading_deg:.2f} degrees"
)

print(
    f"Ground truth displacement: "
    f"{truth_displacement:.2f} meters"
)

print(
    f"Dead reckoning displacement: "
    f"{dr_displacement:.2f} meters"
)

print(
    f"Final position error: "
    f"{final_error:.2f} meters"
)

print(
    f"Drift: "
    f"{drift_percentage:.2f} %"
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)


plt.plot(
    truth_east,
    truth_north,
    label="Vehicle Ground Truth",
    linewidth=2
)


plt.plot(
    east,
    north,
    label="IMU Dead Reckoning V3",
    linewidth=2
)


plt.scatter(
    0,
    0,
    s=60,
    label="Start"
)


plt.xlabel(
    "East (meters)"
)

plt.ylabel(
    "North (meters)"
)

plt.title(
    "GNSS-Denied Dead Reckoning Baseline V3"
)

plt.grid(
    True
)

plt.legend()

plt.axis(
    "equal"
)

plt.tight_layout()

plt.show()