from pathlib import Path
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
MODEL_DIR = ROOT / "models"

smartphone_file = list(DATA_DIR.rglob("S-S1.csv"))[0]
vehicle_file = list(DATA_DIR.rglob("V-S1.csv"))[0]

model_file = MODEL_DIR / "ai_speed_model.joblib"
scaler_file = MODEL_DIR / "ai_speed_scaler.joblib"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

smart = pd.read_csv(
    smartphone_file,
    encoding="cp1252"
)

vehicle = pd.read_csv(
    vehicle_file,
    encoding="cp1252"
)


# ============================================================
# LOAD AI MODEL
# ============================================================

model = joblib.load(model_file)
scaler = joblib.load(scaler_file)

print("AI model loaded successfully.")


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(df, text):

    matches = [
        c for c in df.columns
        if text.lower() in c.lower()
    ]

    if not matches:
        raise ValueError(
            f"Column containing '{text}' not found"
        )

    return matches[0]


# ============================================================
# FIND COLUMNS
# ============================================================

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

time_col = find_column(
    smart,
    "TIME SINCE START"
)

lat_col = find_column(
    smart,
    "GPS LATITUDE"
)

lon_col = find_column(
    smart,
    "GPS LONGITUDE"
)

orientation_yaw_col = find_column(
    smart,
    "ORIENTATION (Yaw)"
)


# ============================================================
# SENSOR ARRAYS
# ============================================================

acc = smart[
    [
        acc_x_col,
        acc_y_col,
        acc_z_col
    ]
].astype(float).values


gyro = smart[
    [
        gyro_yaw_col,
        gyro_pitch_col,
        gyro_roll_col
    ]
].astype(float).values


# IMPORTANT:
# TIME SINCE START is stored in milliseconds.

times_ms = smart[
    time_col
].astype(float).values


# Convert milliseconds → seconds

times = times_ms / 1000.0


gps_lat = smart[
    lat_col
].astype(float).values


gps_lon = smart[
    lon_col
].astype(float).values


orientation_yaw = smart[
    orientation_yaw_col
].astype(float).values


# ============================================================
# VALIDATED OUTAGE
# ============================================================

start_index = 35591
end_index = 36191


indices = np.arange(
    start_index,
    end_index + 1
)


OUTAGE_START = times[start_index]
OUTAGE_END = times[end_index]


print("\n======================================")
print("AI DEAD RECKONING")
print("======================================")

print(
    f"Outage start: {OUTAGE_START:.3f} seconds"
)

print(
    f"Outage end  : {OUTAGE_END:.3f} seconds"
)

print(
    f"Duration     : "
    f"{OUTAGE_END - OUTAGE_START:.3f} seconds"
)

print(
    f"Samples      : {len(indices)}"
)


# ============================================================
# INITIAL POSITION
# ============================================================

start_lat = gps_lat[start_index]
start_lon = gps_lon[start_index]


print(
    f"\nInitial latitude : "
    f"{start_lat:.8f}"
)

print(
    f"Initial longitude: "
    f"{start_lon:.8f}"
)


# ============================================================
# HEADING ALIGNMENT
# ============================================================

# The smartphone orientation yaw is NOT automatically equal
# to the vehicle heading because the phone may be mounted
# at an arbitrary angle.
#
# Therefore we calculate a fixed mounting offset using the
# last known GNSS heading before the outage.
#
# For this experiment, the smartphone GPS orientation is used
# as the GNSS heading reference.

gps_orientation_col = find_column(
    smart,
    "GPS ORIENTATION"
)

gps_orientation = smart[
    gps_orientation_col
].astype(float).values


# Find valid GPS heading measurements before outage

pre_indices = np.where(
    times < OUTAGE_START
)[0]


valid_pre = []

for i in pre_indices[-100:]:

    value = gps_orientation[i]

    if np.isfinite(value):

        valid_pre.append(value)


if valid_pre:

    gnss_heading = float(
        np.median(valid_pre)
    )

else:

    gnss_heading = float(
        gps_orientation[start_index]
    )


# Smartphone orientation immediately before outage

phone_heading = float(
    orientation_yaw[
        pre_indices[-1]
    ]
)


# Calculate mounting offset

heading_offset = (
    gnss_heading
    -
    phone_heading
)


print(
    f"\nPre-outage phone heading : "
    f"{phone_heading:.2f}°"
)

print(
    f"Pre-outage GNSS heading  : "
    f"{gnss_heading:.2f}°"
)

print(
    f"Mounting heading offset  : "
    f"{heading_offset:.2f}°"
)


# ============================================================
# AI SPEED PREDICTION
# ============================================================

WINDOW = 10

predicted_speeds = []


print(
    "\nGenerating AI speed predictions..."
)


for idx in indices:

    if idx < WINDOW:

        if predicted_speeds:

            predicted_speeds.append(
                predicted_speeds[-1]
            )

        else:

            predicted_speeds.append(
                0.0
            )

        continue


    # Last 10 IMU samples

    a = acc[
        idx - WINDOW:idx
    ]

    g = gyro[
        idx - WINDOW:idx
    ]


    features = []


    # Accelerometer features

    for j in range(3):

        features.extend(
            [
                np.mean(a[:, j]),
                np.std(a[:, j]),
                np.min(a[:, j]),
                np.max(a[:, j])
            ]
        )


    # Gyroscope features

    for j in range(3):

        features.extend(
            [
                np.mean(g[:, j]),
                np.std(g[:, j]),
                np.min(g[:, j]),
                np.max(g[:, j])
            ]
        )


    # Sensor magnitudes

    acc_mag = np.linalg.norm(
        a,
        axis=1
    )

    gyro_mag = np.linalg.norm(
        g,
        axis=1
    )


    features.extend(
        [
            np.mean(acc_mag),
            np.std(acc_mag),
            np.min(acc_mag),
            np.max(acc_mag)
        ]
    )


    features.extend(
        [
            np.mean(gyro_mag),
            np.std(gyro_mag),
            np.min(gyro_mag),
            np.max(gyro_mag)
        ]
    )


    features = np.array(
        features
    ).reshape(
        1,
        -1
    )


    scaled_features = scaler.transform(
        features
    )


    prediction = model.predict(
        scaled_features
    )[0]


    prediction = max(
        0.0,
        float(prediction)
    )


    predicted_speeds.append(
        prediction
    )


predicted_speeds = np.array(
    predicted_speeds
)


# ============================================================
# SPEED SMOOTHING
# ============================================================

predicted_speeds = (
    pd.Series(
        predicted_speeds
    )
    .rolling(
        window=5,
        min_periods=1,
        center=True
    )
    .mean()
    .values
)


print("\n======================================")
print("AI SPEED PREDICTION")
print("======================================")

print(
    f"Minimum AI speed : "
    f"{predicted_speeds.min():.2f} km/h"
)

print(
    f"Maximum AI speed : "
    f"{predicted_speeds.max():.2f} km/h"
)

print(
    f"Average AI speed : "
    f"{predicted_speeds.mean():.2f} km/h"
)


# ============================================================
# DEAD RECKONING
# ============================================================

EARTH_RADIUS = 6371000.0


dr_lat = start_lat
dr_lon = start_lon


dr_latitudes = [
    dr_lat
]

dr_longitudes = [
    dr_lon
]


print(
    "\nRunning AI-powered dead reckoning..."
)


for k in range(
    1,
    len(indices)
):

    current_idx = indices[k]

    previous_idx = indices[k - 1]


    # --------------------------------------------------------
    # Correct time interval
    # --------------------------------------------------------

    dt = (
        times[current_idx]
        -
        times[previous_idx]
    )


    if dt <= 0:

        continue


    # --------------------------------------------------------
    # AI predicted speed
    # --------------------------------------------------------

    speed_ms = (
        predicted_speeds[k]
        /
        3.6
    )


    # --------------------------------------------------------
    # Correct smartphone heading
    # --------------------------------------------------------

    raw_heading = (
        orientation_yaw[current_idx]
    )


    heading = (
        raw_heading
        +
        heading_offset
    )


    heading = heading % 360.0


    heading_rad = np.radians(
        heading
    )


    # --------------------------------------------------------
    # North / East velocity
    # --------------------------------------------------------

    north_velocity = (
        speed_ms
        *
        np.cos(heading_rad)
    )


    east_velocity = (
        speed_ms
        *
        np.sin(heading_rad)
    )


    # --------------------------------------------------------
    # Distance during timestep
    # --------------------------------------------------------

    north_distance = (
        north_velocity
        *
        dt
    )


    east_distance = (
        east_velocity
        *
        dt
    )


    # --------------------------------------------------------
    # Update latitude
    # --------------------------------------------------------

    dr_lat += (
        north_distance
        /
        EARTH_RADIUS
    ) * (
        180.0
        /
        np.pi
    )


    # --------------------------------------------------------
    # Update longitude
    # --------------------------------------------------------

    dr_lon += (
        east_distance
        /
        (
            EARTH_RADIUS
            *
            np.cos(
                np.radians(
                    dr_lat
                )
            )
        )
    ) * (
        180.0
        /
        np.pi
    )


    dr_latitudes.append(
        dr_lat
    )

    dr_longitudes.append(
        dr_lon
    )


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def distance_meters(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(
        lat1
    )

    lat2 = np.radians(
        lat2
    )

    dlat = lat2 - lat1

    dlon = np.radians(
        lon2 - lon1
    )


    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )


    return (
        2
        *
        EARTH_RADIUS
        *
        np.arcsin(
            np.sqrt(a)
        )
    )


# ============================================================
# GROUND TRUTH
# ============================================================

truth_lat = gps_lat[
    end_index
]

truth_lon = gps_lon[
    end_index
]


# ============================================================
# POSITION ERROR
# ============================================================

final_error = distance_meters(
    truth_lat,
    truth_lon,
    dr_lat,
    dr_lon
)


ground_truth_displacement = distance_meters(
    start_lat,
    start_lon,
    truth_lat,
    truth_lon
)


# ============================================================
# RESULTS
# ============================================================

print("\n======================================")
print("FINAL AI DEAD RECKONING RESULTS")
print("======================================")

print(
    f"Ground truth latitude : "
    f"{truth_lat:.8f}"
)

print(
    f"Ground truth longitude: "
    f"{truth_lon:.8f}"
)

print(
    f"\nDR latitude           : "
    f"{dr_lat:.8f}"
)

print(
    f"DR longitude          : "
    f"{dr_lon:.8f}"
)

print(
    f"\nGround truth displacement: "
    f"{ground_truth_displacement:.2f} m"
)

print(
    f"Final position error     : "
    f"{final_error:.2f} m"
)


if ground_truth_displacement > 0:

    error_ratio = (
        final_error
        /
        ground_truth_displacement
    ) * 100

    print(
        f"Position error ratio    : "
        f"{error_ratio:.2f}%"
    )


print("\n======================================")
print("AI DEAD RECKONING COMPLETE")
print("======================================")