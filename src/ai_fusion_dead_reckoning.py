# ============================================================
# AI + SENSOR FUSION DEAD RECKONING
# SIH 2026 - GNSS Denied Positioning
#
# Pipeline:
# Smartphone IMU
#       ↓
# Gravity compensation
#       ↓
# Smartphone orientation
#       ↓
# AI speed estimation
#       ↓
# IMU velocity propagation
#       ↓
# AI speed correction
#       ↓
# NHC constraint
#       ↓
# Position integration
#       ↓
# Final dead-reckoned position
# ============================================================


import numpy as np
import pandas as pd
import joblib
import os

from pathlib import Path
from scipy.spatial.transform import Rotation


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Data"

MODEL_DIR = BASE_DIR / "models"


# ============================================================
# 2. FIND DATASETS
# ============================================================

smartphone_file = next(
    DATA_DIR.rglob("S-S1.csv")
)

vehicle_file = next(
    DATA_DIR.rglob("V-S1.csv")
)


# ============================================================
# 3. MODEL FILES
# ============================================================

model_file = (
    MODEL_DIR
    / "ai_speed_model.joblib"
)

scaler_file = (
    MODEL_DIR
    / "ai_speed_scaler.joblib"
)


# ============================================================
# 4. LOAD DATA
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


model = joblib.load(
    model_file
)


scaler = joblib.load(
    scaler_file
)


print(
    "AI model loaded successfully."
)


# ============================================================
# 5. COLUMN FINDER
# ============================================================

def find_column(df, text):

    matches = [
        col
        for col in df.columns
        if text.lower() in col.lower()
    ]

    if not matches:

        raise ValueError(
            f"Column containing '{text}' was not found."
        )

    return matches[0]


# ============================================================
# 6. SMARTPHONE COLUMNS
# ============================================================

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


gps_heading_col = find_column(
    smart,
    "GPS ORIENTATION"
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


orientation_yaw_col = find_column(
    smart,
    "ORIENTATION (Yaw)"
)


orientation_pitch_col = find_column(
    smart,
    "ORIENTATION (Pitch)"
)


orientation_roll_col = find_column(
    smart,
    "ORIENTATION (Roll"
)


# ============================================================
# 7. VEHICLE COLUMNS
# ============================================================

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
# 8. TIME CONVERSION
# ============================================================

# Dataset stores time in milliseconds.
# Convert to seconds.

smart["time_seconds"] = (
    smart[time_col]
    .astype(float)
    / 1000.0
)


# ============================================================
# 9. GNSS OUTAGE
# ============================================================

START_TIME = float(
    os.environ.get(
        "GNSS_START_TIME",
        "3562.0"
    )
)

END_TIME = float(
    os.environ.get(
        "GNSS_END_TIME",
        "3622.0"
    )
)


smart_outage_mask = (

    (smart["time_seconds"] >= START_TIME)

    &

    (smart["time_seconds"] <= END_TIME)

)


smart_outage = smart.loc[
    smart_outage_mask
].copy()


smart_outage = smart_outage.reset_index(
    drop=True
)


# ============================================================
# 10. FIND CORRESPONDING VEHICLE DATA
# ============================================================

start_index = (
    np.abs(
        smart["time_seconds"]
        - START_TIME
    )
).argmin()


end_index = (
    np.abs(
        smart["time_seconds"]
        - END_TIME
    )
).argmin()


vehicle_outage = vehicle.iloc[
    start_index:end_index + 1
].copy()


vehicle_outage = vehicle_outage.reset_index(
    drop=True
)


# ============================================================
# 11. BASIC INFORMATION
# ============================================================

print()
print("======================================")
print("AI + SENSOR FUSION DEAD RECKONING")
print("======================================")


print(
    f"Outage start : "
    f"{smart_outage['time_seconds'].iloc[0]:.3f} seconds"
)


print(
    f"Outage end   : "
    f"{smart_outage['time_seconds'].iloc[-1]:.3f} seconds"
)


print(
    f"Duration     : "
    f"{smart_outage['time_seconds'].iloc[-1] - smart_outage['time_seconds'].iloc[0]:.3f} seconds"
)


print(
    f"Samples      : "
    f"{len(smart_outage)}"
)


# ============================================================
# 12. AI FEATURE STATISTICS
# ============================================================

def statistics(arr):

    return [

        np.mean(arr),

        np.std(arr),

        np.min(arr),

        np.max(arr),

        np.median(arr),

        np.ptp(arr)

    ]


# ============================================================
# 13. AI FEATURE GENERATOR
#
# IMPORTANT:
# This MUST match train_ai_speed_model.py.
#
# Total:
#
# Raw accelerometer       = 18
# Linear acceleration     = 18
# Gyroscope               = 18
# Magnitudes              = 18
# Recent motion           =  6
#                         ----
# Total                   = 78
# ============================================================

def create_features(df, i):

    # --------------------------------------------------------
    # 20 sample window
    # --------------------------------------------------------

    start = max(
        0,
        i - 19
    )


    window = df.iloc[
        start:i + 1
    ]


    # --------------------------------------------------------
    # Raw accelerometer
    # --------------------------------------------------------

    accel_x = (
        window[acc_x_col]
        .astype(float)
        .values
    )


    accel_y = (
        window[acc_y_col]
        .astype(float)
        .values
    )


    accel_z = (
        window[acc_z_col]
        .astype(float)
        .values
    )


    # --------------------------------------------------------
    # Gravity
    # --------------------------------------------------------

    gravity_x = (
        window[gravity_x_col]
        .astype(float)
        .values
    )


    gravity_y = (
        window[gravity_y_col]
        .astype(float)
        .values
    )


    gravity_z = (
        window[gravity_z_col]
        .astype(float)
        .values
    )


    # --------------------------------------------------------
    # Gyroscope
    # --------------------------------------------------------

    gyro_yaw = (
        window[gyro_yaw_col]
        .astype(float)
        .values
    )


    gyro_pitch = (
        window[gyro_pitch_col]
        .astype(float)
        .values
    )


    gyro_roll = (
        window[gyro_roll_col]
        .astype(float)
        .values
    )


    # --------------------------------------------------------
    # Gravity compensated acceleration
    # --------------------------------------------------------

    linear_x = (
        accel_x
        - gravity_x
    )


    linear_y = (
        accel_y
        - gravity_y
    )


    linear_z = (
        accel_z
        - gravity_z
    )


    # --------------------------------------------------------
    # Acceleration magnitude
    # --------------------------------------------------------

    accel_magnitude = np.sqrt(

        accel_x ** 2

        +

        accel_y ** 2

        +

        accel_z ** 2

    )


    # --------------------------------------------------------
    # Linear acceleration magnitude
    # --------------------------------------------------------

    linear_acc_magnitude = np.sqrt(

        linear_x ** 2

        +

        linear_y ** 2

        +

        linear_z ** 2

    )


    # --------------------------------------------------------
    # Gyroscope magnitude
    # --------------------------------------------------------

    gyro_magnitude = np.sqrt(

        gyro_yaw ** 2

        +

        gyro_pitch ** 2

        +

        gyro_roll ** 2

    )


    features = []


    # ========================================================
    # RAW ACCELEROMETER
    # ========================================================

    for arr in [

        accel_x,

        accel_y,

        accel_z

    ]:

        features.extend(
            statistics(arr)
        )


    # ========================================================
    # LINEAR ACCELERATION
    # ========================================================

    for arr in [

        linear_x,

        linear_y,

        linear_z

    ]:

        features.extend(
            statistics(arr)
        )


    # ========================================================
    # GYROSCOPE
    # ========================================================

    for arr in [

        gyro_yaw,

        gyro_pitch,

        gyro_roll

    ]:

        features.extend(
            statistics(arr)
        )


    # ========================================================
    # MAGNITUDES
    # ========================================================

    for arr in [

        accel_magnitude,

        linear_acc_magnitude,

        gyro_magnitude

    ]:

        features.extend(
            statistics(arr)
        )


    # ========================================================
    # RECENT MOTION FEATURES
    # ========================================================

    last_5 = slice(

        max(
            0,
            len(window) - 5
        ),

        len(window)

    )


    last_10 = slice(

        max(
            0,
            len(window) - 10
        ),

        len(window)

    )


    features.extend([

        np.mean(
            linear_acc_magnitude[last_5]
        ),

        np.std(
            linear_acc_magnitude[last_5]
        ),

        np.mean(
            linear_acc_magnitude[last_10]
        ),

        np.std(
            linear_acc_magnitude[last_10]
        ),

        np.mean(
            gyro_magnitude[last_5]
        ),

        np.std(
            gyro_magnitude[last_5]
        )

    ])


    # ========================================================
    # FINAL FEATURE VECTOR
    # ========================================================

    features = np.asarray(
        features,
        dtype=np.float64
    )


    # Safety check

    if len(features) != 78:

        raise ValueError(
            f"Expected 78 features, "
            f"but generated {len(features)}."
        )


    return features


# ============================================================
# 14. GENERATE AI SPEED PREDICTIONS
# ============================================================

print()
print(
    "Generating AI speed predictions..."
)


# ------------------------------------------------------------
# IMPORTANT:
#
# The AI was trained with a 20-sample window.
#
# At 10 Hz:
#
# 20 samples ≈ 2 seconds
#
# Therefore we provide the AI with the last 2 seconds
# BEFORE the GNSS outage.
#
# This history contains smartphone IMU only.
# No future vehicle information is used.
# ------------------------------------------------------------

history_mask = (

    (smart["time_seconds"] >= START_TIME - 2.0)

    &

    (smart["time_seconds"] < START_TIME)

)


pre_outage_history = smart.loc[
    history_mask
].copy()


pre_outage_history = pre_outage_history.reset_index(
    drop=True
)


# ------------------------------------------------------------
# Combine:
#
# pre-outage smartphone history
# +
# outage smartphone data
# ------------------------------------------------------------

combined_for_ai = pd.concat(

    [

        pre_outage_history,

        smart_outage

    ],

    ignore_index=True

)


outage_start_position = (
    len(pre_outage_history)
)


ai_speeds = []


# ------------------------------------------------------------
# Predict speed for each outage sample.
# ------------------------------------------------------------

for i in range(

    outage_start_position,

    len(combined_for_ai)

):

    features = create_features(
        combined_for_ai,
        i
    )


    scaled_features = scaler.transform(

        features.reshape(
            1,
            -1
        )

    )


    prediction = model.predict(
        scaled_features
    )[0]


    ai_speeds.append(
        prediction
    )


ai_speeds = np.asarray(
    ai_speeds
)


# ------------------------------------------------------------
# Prevent negative speed.
# ------------------------------------------------------------

ai_speeds = np.clip(

    ai_speeds,

    0,

    None

)


print()
print("======================================")
print("AI SPEED")
print("======================================")


print(
    f"Minimum : "
    f"{ai_speeds.min():.2f} km/h"
)


print(
    f"Maximum : "
    f"{ai_speeds.max():.2f} km/h"
)


print(
    f"Average : "
    f"{ai_speeds.mean():.2f} km/h"
)


# ============================================================
# 15. CIRCULAR MEAN FOR ANGLES
# ============================================================

def circular_mean_deg(
    angles
):

    angles = np.asarray(
        angles,
        dtype=float
    )


    angles_rad = np.deg2rad(
        angles
    )


    sin_mean = np.mean(
        np.sin(angles_rad)
    )


    cos_mean = np.mean(
        np.cos(angles_rad)
    )


    result = np.rad2deg(

        np.arctan2(

            sin_mean,

            cos_mean

        )

    )


    return result % 360


# ============================================================
# 16. HEADING INITIALIZATION
# ============================================================

phone_heading = (

    smart_outage[
        orientation_yaw_col
    ]

    .astype(float)

    .values

)


gnss_heading = (

    smart_outage[
        gps_heading_col
    ]

    .astype(float)

    .values

)


# ------------------------------------------------------------
# Use first five samples instead of one sample.
# ------------------------------------------------------------

phone_heading_initial = circular_mean_deg(

    phone_heading[:5]

)


gnss_heading_initial = circular_mean_deg(

    gnss_heading[:5]

)


# ------------------------------------------------------------
# Shortest angular difference.
# ------------------------------------------------------------

heading_offset = (

    gnss_heading_initial

    -

    phone_heading_initial

    +

    180

) % 360 - 180


# ------------------------------------------------------------
# Align phone orientation to GNSS heading frame.
# ------------------------------------------------------------

aligned_heading = (

    phone_heading

    +

    heading_offset

) % 360


# ------------------------------------------------------------
# Stable initial heading.
# ------------------------------------------------------------

initial_heading = circular_mean_deg(

    aligned_heading[:5]

)


print()
print("======================================")
print("HEADING INITIALIZATION")
print("======================================")


print(
    f"Phone heading : "
    f"{phone_heading_initial:.2f}°"
)


print(
    f"GNSS heading  : "
    f"{gnss_heading_initial:.2f}°"
)


print(
    f"Heading offset: "
    f"{heading_offset:.2f}°"
)


print(
    f"Initial heading: "
    f"{initial_heading:.2f}°"
)


# ============================================================
# 17. INITIAL POSITION
# ============================================================

initial_lat = float(

    smart_outage[
        lat_col
    ].iloc[0]

)


initial_lon = float(

    smart_outage[
        lon_col
    ].iloc[0]

)


print()
print("======================================")
print("INITIAL NAVIGATION STATE")
print("======================================")


print(
    f"Initial latitude : "
    f"{initial_lat:.8f}"
)


print(
    f"Initial longitude: "
    f"{initial_lon:.8f}"
)


# ============================================================
# 18. INITIAL AI SPEED
# ============================================================

# Use median of first 5 predictions.
#
# This is more stable than using only ai_speeds[0].

initial_speed = (

    np.median(
        ai_speeds[:5]
    )

    / 3.6

)


initial_heading_rad = np.deg2rad(

    initial_heading

)


# ------------------------------------------------------------
# Heading convention:
#
# 0°   = North
# 90°  = East
# 180° = South
# 270° = West
# ------------------------------------------------------------

initial_east_velocity = (

    initial_speed

    *

    np.sin(
        initial_heading_rad
    )

)


initial_north_velocity = (

    initial_speed

    *

    np.cos(
        initial_heading_rad
    )

)


print(
    f"Initial AI speed : "
    f"{initial_speed * 3.6:.2f} km/h"
)


print(
    f"Initial East velocity : "
    f"{initial_east_velocity:.3f} m/s"
)


print(
    f"Initial North velocity: "
    f"{initial_north_velocity:.3f} m/s"
)


# ============================================================
# 19. EARTH CONSTANT
# ============================================================

EARTH_RADIUS = 6371000.0


# ============================================================
# 20. POSITION + VELOCITY STATE
# ============================================================

east_position = 0.0

north_position = 0.0


east_velocity = (
    initial_east_velocity
)

north_velocity = (
    initial_north_velocity
)


# ============================================================
# 21. SENSOR ARRAYS
# ============================================================

acc_x = (

    smart_outage[
        acc_x_col
    ]

    .astype(float)

    .values

)


acc_y = (

    smart_outage[
        acc_y_col
    ]

    .astype(float)

    .values

)


acc_z = (

    smart_outage[
        acc_z_col
    ]

    .astype(float)

    .values

)


gravity_x = (

    smart_outage[
        gravity_x_col
    ]

    .astype(float)

    .values

)


gravity_y = (

    smart_outage[
        gravity_y_col
    ]

    .astype(float)

    .values

)


gravity_z = (

    smart_outage[
        gravity_z_col
    ]

    .astype(float)

    .values

)


pitch = (

    smart_outage[
        orientation_pitch_col
    ]

    .astype(float)

    .values

)


roll = (

    smart_outage[
        orientation_roll_col
    ]

    .astype(float)

    .values

)


yaw = (

    smart_outage[
        orientation_yaw_col
    ]

    .astype(float)

    .values

)


times = (

    smart_outage[
        "time_seconds"
    ]

    .astype(float)

    .values

)


# ============================================================
# 22. RESULT STORAGE
# ============================================================

estimated_latitudes = [

    initial_lat

]


estimated_longitudes = [

    initial_lon

]


estimated_speeds = [

    initial_speed * 3.6

]


# ============================================================
# 23. MAIN SENSOR FUSION LOOP
# ============================================================

print()
print(
    "Running AI + IMU + NHC fusion..."
)


for i in range(

    1,

    len(smart_outage)

):


    # ========================================================
    # TIME STEP
    # ========================================================

    dt = (

        times[i]

        -

        times[i - 1]

    )


    # Safety check.

    if (

        dt <= 0

        or

        dt > 0.5

    ):

        dt = 0.1


    # ========================================================
    # GRAVITY COMPENSATION
    # ========================================================

    linear_x = (

        acc_x[i]

        -

        gravity_x[i]

    )


    linear_y = (

        acc_y[i]

        -

        gravity_y[i]

    )


    linear_z = (

        acc_z[i]

        -

        gravity_z[i]

    )


    # ========================================================
    # ROTATE PHONE ACCELERATION
    # TO WORLD COORDINATES
    # ========================================================

    try:

        rotation = Rotation.from_euler(

            "ZYX",

            [

                yaw[i],

                pitch[i],

                roll[i]

            ],

            degrees=True

        )


        world_acceleration = rotation.apply(

            [

                linear_x,

                linear_y,

                linear_z

            ]

        )


    except Exception:

        world_acceleration = np.array(

            [

                linear_x,

                linear_y,

                linear_z

            ]

        )


    # ========================================================
    # WORLD COORDINATE SYSTEM
    #
    # X -> East
    # Y -> North
    # Z -> Up
    # ========================================================

    measured_east_acceleration = (

        world_acceleration[0]

    )


    measured_north_acceleration = (

        world_acceleration[1]

    )


    # ========================================================
    # LIMIT EXTREME SENSOR SPIKES
    # ========================================================

    measured_east_acceleration = np.clip(

        measured_east_acceleration,

        -8.0,

        8.0

    )


    measured_north_acceleration = np.clip(

        measured_north_acceleration,

        -8.0,

        8.0

    )


    # ========================================================
    # AI SPEED
    # ========================================================

    ai_speed = (

        ai_speeds[i]

        /

        3.6

    )


    # ========================================================
    # IMU VELOCITY PREDICTION
    # ========================================================

    predicted_east_velocity = (

        east_velocity

        +

        measured_east_acceleration

        *

        dt

    )


    predicted_north_velocity = (

        north_velocity

        +

        measured_north_acceleration

        *

        dt

    )


    # ========================================================
    # PREDICTED SPEED FROM IMU
    # ========================================================

    predicted_speed = np.sqrt(

        predicted_east_velocity ** 2

        +

        predicted_north_velocity ** 2

    )


    # ========================================================
    # COMBINE IMU DIRECTION WITH AI SPEED
    # ========================================================

    if predicted_speed > 0.1:

        direction_east = (

            predicted_east_velocity

            /

            predicted_speed

        )


        direction_north = (

            predicted_north_velocity

            /

            predicted_speed

        )


        ai_east_velocity = (

            direction_east

            *

            ai_speed

        )


        ai_north_velocity = (

            direction_north

            *

            ai_speed

        )


    else:

        heading_rad = np.deg2rad(

            aligned_heading[i]

        )


        ai_east_velocity = (

            ai_speed

            *

            np.sin(
                heading_rad
            )

        )


        ai_north_velocity = (

            ai_speed

            *

            np.cos(
                heading_rad
            )

        )


    # ========================================================
    # AI + IMU FUSION
    # ========================================================

    imu_weight = 0.65

    ai_weight = 0.35


    east_velocity = (

        imu_weight

        *

        predicted_east_velocity

        +

        ai_weight

        *

        ai_east_velocity

    )


    north_velocity = (

        imu_weight

        *

        predicted_north_velocity

        +

        ai_weight

        *

        ai_north_velocity

    )


    # ========================================================
    # NON-HOLONOMIC CONSTRAINT
    #
    # Vehicle normally does not move sideways
    # like a crab.
    #
    # Apply a soft correction.
    # ========================================================

    current_speed = np.sqrt(

        east_velocity ** 2

        +

        north_velocity ** 2

    )


    if current_speed > 1.0:

        heading_rad = np.deg2rad(

            aligned_heading[i]

        )


        forward_east = np.sin(

            heading_rad

        )


        forward_north = np.cos(

            heading_rad

        )


        # ----------------------------------------------------
        # Forward component.
        # ----------------------------------------------------

        forward_velocity = (

            east_velocity

            *

            forward_east

            +

            north_velocity

            *

            forward_north

        )


        # ----------------------------------------------------
        # Lateral component.
        # ----------------------------------------------------

        lateral_east = (

            east_velocity

            -

            forward_velocity

            *

            forward_east

        )


        lateral_north = (

            north_velocity

            -

            forward_velocity

            *

            forward_north

        )


        # ----------------------------------------------------
        # Soft NHC correction.
        # ----------------------------------------------------

        nhc_weight = 0.15


        east_velocity -= (

            nhc_weight

            *

            lateral_east

        )


        north_velocity -= (

            nhc_weight

            *

            lateral_north

        )


    # ========================================================
    # POSITION INTEGRATION
    # ========================================================

    east_position += (

        east_velocity

        *

        dt

    )


    north_position += (

        north_velocity

        *

        dt

    )


    # ========================================================
    # LOCAL METERS -> LATITUDE
    # ========================================================

    latitude = (

        initial_lat

        +

        np.rad2deg(

            north_position

            /

            EARTH_RADIUS

        )

    )


    # ========================================================
    # LOCAL METERS -> LONGITUDE
    # ========================================================

    longitude = (

        initial_lon

        +

        np.rad2deg(

            east_position

            /

            (

                EARTH_RADIUS

                *

                np.cos(

                    np.deg2rad(

                        initial_lat

                    )

                )

            )

        )

    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    estimated_latitudes.append(

        latitude

    )


    estimated_longitudes.append(

        longitude

    )


    estimated_speeds.append(

        np.sqrt(

            east_velocity ** 2

            +

            north_velocity ** 2

        )

        *

        3.6

    )


# ============================================================
# 24. FINAL POSITION
# ============================================================

final_latitude = (

    estimated_latitudes[-1]

)


final_longitude = (

    estimated_longitudes[-1]

)


# ============================================================
# 25. GROUND TRUTH POSITION
# ============================================================

ground_truth_latitude = float(

    smart_outage[
        lat_col
    ].iloc[-1]

)


ground_truth_longitude = float(

    smart_outage[
        lon_col
    ].iloc[-1]

)


# ============================================================
# 26. HAVERSINE FUNCTION
# ============================================================

def haversine_distance(

    lat1,

    lon1,

    lat2,

    lon2

):

    lat1_rad = np.deg2rad(
        lat1
    )


    lat2_rad = np.deg2rad(
        lat2
    )


    dlat = (

        lat2_rad

        -

        lat1_rad

    )


    dlon = np.deg2rad(

        lon2

        -

        lon1

    )


    a = (

        np.sin(
            dlat / 2
        ) ** 2

        +

        np.cos(
            lat1_rad
        )

        *

        np.cos(
            lat2_rad
        )

        *

        np.sin(
            dlon / 2
        ) ** 2

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
# 27. FINAL POSITION ERROR
# ============================================================

final_position_error = haversine_distance(

    final_latitude,

    final_longitude,

    ground_truth_latitude,

    ground_truth_longitude

)


# ============================================================
# 28. GROUND TRUTH DISPLACEMENT
# ============================================================

ground_truth_displacement = haversine_distance(

    initial_lat,

    initial_lon,

    ground_truth_latitude,

    ground_truth_longitude

)


# ============================================================
# 29. GROUND TRUTH TRAVEL DISTANCE
# ============================================================

vehicle_latitudes = (

    vehicle_outage[
        vehicle_lat_col
    ]

    .astype(float)

    .values

)


vehicle_longitudes = (

    vehicle_outage[
        vehicle_lon_col
    ]

    .astype(float)

    .values

)


ground_truth_travel_distance = 0.0


for i in range(

    1,

    len(vehicle_outage)

):

    ground_truth_travel_distance += (

        haversine_distance(

            vehicle_latitudes[i - 1],

            vehicle_longitudes[i - 1],

            vehicle_latitudes[i],

            vehicle_longitudes[i]

        )

    )


# ============================================================
# 30. ERROR RATIO
# ============================================================

if ground_truth_travel_distance > 0:

    error_ratio = (

        final_position_error

        /

        ground_truth_travel_distance

    ) * 100

else:

    error_ratio = 0.0


# ============================================================
# 31. SPEED RESULTS
# ============================================================

final_fused_speed = (

    estimated_speeds[-1]

)


average_fused_speed = np.mean(

    estimated_speeds

)


# ============================================================
# 32. FINAL RESULTS
# ============================================================

print()
print("======================================")
print("FINAL RESULTS")
print("======================================")


print()
print("GROUND TRUTH")


print(
    f"Latitude  : "
    f"{ground_truth_latitude:.8f}"
)


print(
    f"Longitude : "
    f"{ground_truth_longitude:.8f}"
)


print()
print("FUSED DEAD RECKONING")


print(
    f"Latitude  : "
    f"{final_latitude:.8f}"
)


print(
    f"Longitude : "
    f"{final_longitude:.8f}"
)


print()
print(
    f"Ground truth displacement : "
    f"{ground_truth_displacement:.2f} m"
)


print(
    f"Ground truth traveled distance : "
    f"{ground_truth_travel_distance:.2f} m"
)


print(
    f"Final position error      : "
    f"{final_position_error:.2f} m"
)


print(
    f"Position error ratio      : "
    f"{error_ratio:.2f}%"
)


print(
    f"Final fused speed         : "
    f"{final_fused_speed:.2f} km/h"
)


print(
    f"Average fused speed       : "
    f"{average_fused_speed:.2f} km/h"
)


# ============================================================
# 33. SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "time_seconds":
        smart_outage[
            "time_seconds"
        ].values,

    "latitude":
        estimated_latitudes,

    "longitude":
        estimated_longitudes,

    "ai_speed_kmh":
        ai_speeds,

    "fused_speed_kmh":
        estimated_speeds,

    "aligned_heading_deg":
        aligned_heading

})


results_file = (

    BASE_DIR

    /

    "ai_fusion_results.csv"

)


results.to_csv(

    results_file,

    index=False

)


print()
print(
    "Results saved to:"
)


print(
    results_file
)


print()
print("======================================")
print("AI + IMU + NHC FUSION COMPLETE")
print("======================================")