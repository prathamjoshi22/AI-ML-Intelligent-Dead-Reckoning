# ============================================================
# SIH 2026 - AI + IMU DEAD RECKONING EVALUATION
#
# Purpose:
#   Compare our fused dead-reckoning trajectory against the
#   verified vehicle ground-truth trajectory.
#
# IMPORTANT:
#   The vehicle trajectory is shifted so that its starting
#   point is the same as the DR starting point.
#
#   This removes constant GPS position offset and measures
#   actual dead-reckoning drift.
# ============================================================

import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Data"

RESULT_FILE = BASE_DIR / "src" / "ai_fusion_results.csv"


# ============================================================
# 2. FIND VEHICLE DATASET
# ============================================================

vehicle_file = next(
    DATA_DIR.rglob("V-S1.csv")
)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("Loading vehicle ground truth...")

vehicle = pd.read_csv(
    vehicle_file,
    encoding="cp1252"
)


print(
    f"Vehicle rows: {len(vehicle)}"
)


# ============================================================
# 4. LOAD DR RESULTS
# ============================================================

print()

print("Loading AI + IMU fusion results...")

if not RESULT_FILE.exists():

    raise FileNotFoundError(
        f"Could not find:\n{RESULT_FILE}\n\n"
        "Run ai_fusion_dead_reckoning.py first."
    )


dr = pd.read_csv(
    RESULT_FILE
)


print(
    f"DR rows: {len(dr)}"
)


print()

print("DR columns:")

for col in dr.columns:

    print(
        f"  {col}"
    )


# ============================================================
# 5. COLUMN FINDER
# ============================================================

def find_column(df, possible_names):

    for name in possible_names:

        for col in df.columns:

            if name.lower() == col.lower():

                return col


    for name in possible_names:

        for col in df.columns:

            if name.lower() in col.lower():

                return col


    raise ValueError(
        "Could not find required column.\n"
        f"Tried: {possible_names}"
    )


# ============================================================
# 6. FIND DR LAT/LON
# ============================================================

dr_lat_col = find_column(
    dr,
    [
        "latitude",
        "fused_latitude",
        "dr_latitude",
        "lat"
    ]
)


dr_lon_col = find_column(
    dr,
    [
        "longitude",
        "fused_longitude",
        "dr_longitude",
        "lon"
    ]
)


print()

print(
    f"DR latitude column : {dr_lat_col}"
)

print(
    f"DR longitude column: {dr_lon_col}"
)


# ============================================================
# 7. FIND VEHICLE COLUMNS
# ============================================================

vehicle_lat_col = find_column(
    vehicle,
    [
        "Latitude"
    ]
)


vehicle_lon_col = find_column(
    vehicle,
    [
        "Longitude"
    ]
)


# ============================================================
# 8. OUTAGE ROWS
# ============================================================

START_INDEX = 35591

END_INDEX = 36191


vehicle_segment = vehicle.iloc[
    START_INDEX:
    END_INDEX + 1
].copy()


vehicle_segment.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# 9. EXTRACT VEHICLE GPS
# ============================================================

vehicle_lat = (

    vehicle_segment[
        vehicle_lat_col
    ]

    .astype(float)

    .values

)


vehicle_lon = (

    vehicle_segment[
        vehicle_lon_col
    ]

    .astype(float)

    .values

)


# ============================================================
# 10. EXTRACT DR GPS
# ============================================================

dr_lat = (

    dr[
        dr_lat_col
    ]

    .astype(float)

    .values

)


dr_lon = (

    dr[
        dr_lon_col
    ]

    .astype(float)

    .values

)


# ============================================================
# 11. MAKE LENGTHS MATCH
# ============================================================

n = min(

    len(vehicle_lat),
    len(dr_lat)

)


vehicle_lat = vehicle_lat[:n]

vehicle_lon = vehicle_lon[:n]

dr_lat = dr_lat[:n]

dr_lon = dr_lon[:n]


print()

print(
    f"Evaluation samples: {n}"
)


# ============================================================
# 12. LOCAL COORDINATE CONVERSION
# ============================================================

EARTH_RADIUS = 6371000.0


def gps_to_local(

    lat,
    lon,
    reference_lat,
    reference_lon

):

    latitude_scale = (

        np.pi
        *
        EARTH_RADIUS
        /
        180.0

    )


    longitude_scale = (

        latitude_scale
        *
        np.cos(
            np.deg2rad(
                reference_lat
            )
        )

    )


    east = (

        (lon - reference_lon)
        *
        longitude_scale

    )


    north = (

        (lat - reference_lat)
        *
        latitude_scale

    )


    return east, north


# ============================================================
# 13. VEHICLE GROUND TRUTH
# ============================================================

vehicle_start_lat = vehicle_lat[0]

vehicle_start_lon = vehicle_lon[0]


vehicle_east, vehicle_north = gps_to_local(

    vehicle_lat,
    vehicle_lon,

    vehicle_start_lat,
    vehicle_start_lon

)


# ============================================================
# 14. DR TRAJECTORY
# ============================================================

dr_start_lat = dr_lat[0]

dr_start_lon = dr_lon[0]


dr_east, dr_north = gps_to_local(

    dr_lat,
    dr_lon,

    dr_start_lat,
    dr_start_lon

)


# ============================================================
# 15. ALIGN GROUND TRUTH TO DR START
# ============================================================

#
# Both trajectories now start at:
#
# East  = 0
# North = 0
#
# Therefore we compare only movement/drift.
#


# ============================================================
# 16. FINAL POSITIONS
# ============================================================

truth_final_east = vehicle_east[-1]

truth_final_north = vehicle_north[-1]


dr_final_east = dr_east[-1]

dr_final_north = dr_north[-1]


# ============================================================
# 17. FINAL DRIFT ERROR
# ============================================================

final_error = np.sqrt(

    (dr_final_east - truth_final_east) ** 2

    +

    (dr_final_north - truth_final_north) ** 2

)


# ============================================================
# 18. GROUND TRUTH ENDPOINT DISTANCE
# ============================================================

truth_endpoint_distance = np.sqrt(

    truth_final_east ** 2

    +

    truth_final_north ** 2

)


# ============================================================
# 19. DR ENDPOINT DISTANCE
# ============================================================

dr_endpoint_distance = np.sqrt(

    dr_final_east ** 2

    +

    dr_final_north ** 2

)


# ============================================================
# 20. DRIFT PERCENTAGE
# ============================================================

#
# SIH-style drift:
#
# final position error / traveled distance
#


# Calculate vehicle traveled distance.

truth_traveled_distance = 0.0


for i in range(

    1,
    n

):

    truth_traveled_distance += np.sqrt(

        (
            vehicle_east[i]
            -
            vehicle_east[i - 1]
        ) ** 2

        +

        (
            vehicle_north[i]
            -
            vehicle_north[i - 1]
        ) ** 2

    )


drift_percentage = (

    final_error
    /
    truth_traveled_distance
    *
    100.0

)


# ============================================================
# 21. TRAJECTORY RMSE
# ============================================================

trajectory_error = np.sqrt(

    (
        dr_east
        -
        vehicle_east
    ) ** 2

    +

    (
        dr_north
        -
        vehicle_north
    ) ** 2

)


trajectory_rmse = np.sqrt(

    np.mean(
        trajectory_error ** 2
    )

)


trajectory_mean_error = np.mean(
    trajectory_error
)


trajectory_max_error = np.max(
    trajectory_error
)


# ============================================================
# 22. FINAL HEADING OF TRAJECTORIES
# ============================================================

truth_heading = np.degrees(

    np.arctan2(

        truth_final_east,
        truth_final_north

    )

)


dr_heading = np.degrees(

    np.arctan2(

        dr_final_east,
        dr_final_north

    )

)


truth_heading = (
    truth_heading + 360
) % 360


dr_heading = (
    dr_heading + 360
) % 360


# ============================================================
# 23. PRINT RESULTS
# ============================================================

print()

print(
    "=================================================="
)

print(
    "PROPER AI + IMU DEAD RECKONING EVALUATION"
)

print(
    "=================================================="
)


print()

print("GROUND TRUTH")

print(
    f"Final East displacement  : "
    f"{truth_final_east:.2f} m"
)

print(
    f"Final North displacement : "
    f"{truth_final_north:.2f} m"
)

print(
    f"Endpoint displacement    : "
    f"{truth_endpoint_distance:.2f} m"
)

print(
    f"Traveled distance        : "
    f"{truth_traveled_distance:.2f} m"
)


print()

print("DEAD RECKONING")

print(
    f"Final East displacement  : "
    f"{dr_final_east:.2f} m"
)

print(
    f"Final North displacement : "
    f"{dr_final_north:.2f} m"
)

print(
    f"DR endpoint displacement : "
    f"{dr_endpoint_distance:.2f} m"
)


print()

print("ERROR")

print(
    f"Final position error     : "
    f"{final_error:.2f} m"
)

print(
    f"Drift percentage         : "
    f"{drift_percentage:.2f}%"
)

print(
    f"Trajectory RMSE          : "
    f"{trajectory_rmse:.2f} m"
)

print(
    f"Average trajectory error : "
    f"{trajectory_mean_error:.2f} m"
)

print(
    f"Maximum trajectory error : "
    f"{trajectory_max_error:.2f} m"
)


print()

print("HEADING")

print(
    f"Ground truth final heading : "
    f"{truth_heading:.2f}°"
)

print(
    f"DR final heading           : "
    f"{dr_heading:.2f}°"
)


# ============================================================
# 24. BENCHMARK CHECK
# ============================================================

print()

print(
    "=================================================="
)

print(
    "SIH BENCHMARK CHECK"
)

print(
    "=================================================="
)


if drift_percentage < 10:

    print(
        "PASS: DR drift is below 10%."
    )

else:

    print(
        "CURRENT RESULT: "
        f"{drift_percentage:.2f}% drift."
    )

    print(
        "Target: <10% drift."
    )


# ============================================================
# 25. SAVE COMPARISON DATA
# ============================================================

comparison = pd.DataFrame({

    "vehicle_east_m":
        vehicle_east,

    "vehicle_north_m":
        vehicle_north,

    "dr_east_m":
        dr_east,

    "dr_north_m":
        dr_north,

    "position_error_m":
        trajectory_error

})


comparison_file = (

    BASE_DIR
    /
    "fusion_evaluation.csv"

)


comparison.to_csv(

    comparison_file,
    index=False

)


print()

print(
    "Comparison saved to:"
)

print(
    comparison_file
)


print()

print(
    "=================================================="
)

print(
    "EVALUATION COMPLETE"
)

print(
    "=================================================="
)