# ============================================================
# VERIFY GROUND TRUTH FOR GNSS-DENIED SEGMENT
# SIH 2026 - GNSS Dead Reckoning
#
# Purpose:
#   Verify the vehicle reference trajectory used as
#   ground truth before evaluating the AI + IMU system.
#
# IMPORTANT:
#   Vehicle data is treated as ground truth.
#   Smartphone GPS is NOT used for final benchmark.
# ============================================================

import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Data"


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
# 3. LOAD DATA
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


print(
    f"Smartphone rows: {len(smart)}"
)

print(
    f"Vehicle rows   : {len(vehicle)}"
)


# ============================================================
# 4. COLUMN FINDER
# ============================================================

def find_column(df, text):

    matches = [

        col

        for col in df.columns

        if text.lower() in col.lower()

    ]

    if not matches:

        raise ValueError(
            f"Column containing '{text}' not found."
        )

    return matches[0]


# ============================================================
# 5. SMARTPHONE TIME
# ============================================================

smart_time_col = find_column(
    smart,
    "TIME SINCE START"
)


smart["time_seconds"] = (

    smart[
        smart_time_col
    ]

    .astype(float)

    / 1000.0

)


# ============================================================
# 6. VEHICLE COLUMNS
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
# 7. OUTAGE
# ============================================================

START_TIME = 3562.0

END_TIME = 3622.0


# ============================================================
# 8. FIND EXACT SMARTPHONE ROWS
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


actual_start_time = float(

    smart[
        "time_seconds"
    ].iloc[start_index]

)


actual_end_time = float(

    smart[
        "time_seconds"
    ].iloc[end_index]

)


# ============================================================
# 9. VEHICLE ROW-WISE MATCHING
# ============================================================

vehicle_segment = vehicle.iloc[

    start_index:
    end_index + 1

].copy()


vehicle_segment = (

    vehicle_segment

    .reset_index(drop=True)

)


# ============================================================
# 10. VEHICLE GPS ARRAYS
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


vehicle_speed = (

    vehicle_segment[
        vehicle_speed_col
    ]

    .astype(float)

    .values

)


vehicle_heading = (

    vehicle_segment[
        vehicle_heading_col
    ]

    .astype(float)

    .values

)


# ============================================================
# 11. HAVERSINE DISTANCE
# ============================================================

EARTH_RADIUS = 6371000.0


def haversine(

    lat1,
    lon1,
    lat2,
    lon2

):

    lat1 = np.deg2rad(lat1)

    lat2 = np.deg2rad(lat2)

    dlat = lat2 - lat1

    dlon = np.deg2rad(
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
# 12. GROUND TRUTH START / END
# ============================================================

start_lat = vehicle_lat[0]

start_lon = vehicle_lon[0]

end_lat = vehicle_lat[-1]

end_lon = vehicle_lon[-1]


# ============================================================
# 13. ENDPOINT DISPLACEMENT
# ============================================================

endpoint_distance = haversine(

    start_lat,
    start_lon,

    end_lat,
    end_lon

)


# ============================================================
# 14. TRAVELED GPS DISTANCE
# ============================================================

gps_traveled_distance = 0.0


for i in range(

    1,
    len(vehicle_segment)

):

    gps_traveled_distance += haversine(

        vehicle_lat[i - 1],
        vehicle_lon[i - 1],

        vehicle_lat[i],
        vehicle_lon[i]

    )


# ============================================================
# 15. SPEED-BASED DISTANCE
# ============================================================

#
# Vehicle data has a sample period of approximately
# 0.1 seconds.
#

speed_distance = 0.0


for i in range(

    1,
    len(vehicle_segment)

):

    dt = 0.1


    speed_mps = (

        vehicle_speed[i]

        /

        3.6

    )


    speed_distance += (

        speed_mps
        *
        dt

    )


# ============================================================
# 16. AVERAGE SPEED DISTANCE
# ============================================================

duration = (

    actual_end_time
    -
    actual_start_time

)


average_speed = np.mean(
    vehicle_speed
)


average_speed_distance = (

    average_speed
    /
    3.6
    *
    duration

)


# ============================================================
# 17. LOCAL EAST / NORTH DISPLACEMENT
# ============================================================

# Convert vehicle endpoint to local meters relative
# to the vehicle starting point.

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
        np.deg2rad(start_lat)
    )

)


north_displacement = (

    (end_lat - start_lat)
    *
    latitude_scale

)


east_displacement = (

    (end_lon - start_lon)
    *
    longitude_scale

)


local_displacement = np.sqrt(

    east_displacement ** 2

    +

    north_displacement ** 2

)


# ============================================================
# 18. PRINT RESULTS
# ============================================================

print()
print("==============================================")
print("GROUND TRUTH VERIFICATION")
print("==============================================")


print()
print("SEGMENT")

print(
    f"Requested start : "
    f"{START_TIME:.3f} s"
)

print(
    f"Actual start    : "
    f"{actual_start_time:.3f} s"
)

print(
    f"Actual end      : "
    f"{actual_end_time:.3f} s"
)

print(
    f"Duration        : "
    f"{duration:.3f} s"
)

print(
    f"Samples         : "
    f"{len(vehicle_segment)}"
)


print()
print("VEHICLE START")

print(
    f"Latitude  : "
    f"{start_lat:.10f}"
)

print(
    f"Longitude : "
    f"{start_lon:.10f}"
)


print()
print("VEHICLE END")

print(
    f"Latitude  : "
    f"{end_lat:.10f}"
)

print(
    f"Longitude : "
    f"{end_lon:.10f}"
)


print()
print("GROUND TRUTH DISPLACEMENT")

print(
    f"East  : "
    f"{east_displacement:.2f} m"
)

print(
    f"North : "
    f"{north_displacement:.2f} m"
)

print(
    f"Endpoint displacement : "
    f"{endpoint_distance:.2f} m"
)


print()
print("GROUND TRUTH DISTANCE")

print(
    f"GPS traveled distance : "
    f"{gps_traveled_distance:.2f} m"
)

print(
    f"Speed-based distance  : "
    f"{speed_distance:.2f} m"
)

print(
    f"Average-speed distance: "
    f"{average_speed_distance:.2f} m"
)


print()
print("VEHICLE MOTION")

print(
    f"Minimum speed : "
    f"{vehicle_speed.min():.2f} km/h"
)

print(
    f"Maximum speed : "
    f"{vehicle_speed.max():.2f} km/h"
)

print(
    f"Average speed : "
    f"{average_speed:.2f} km/h"
)

print(
    f"Initial speed : "
    f"{vehicle_speed[0]:.2f} km/h"
)

print(
    f"Final speed   : "
    f"{vehicle_speed[-1]:.2f} km/h"
)


print()
print("HEADING")

print(
    f"Initial heading : "
    f"{vehicle_heading[0]:.2f}°"
)

print(
    f"Final heading   : "
    f"{vehicle_heading[-1]:.2f}°"
)

print(
    f"Average heading : "
    f"{np.mean(vehicle_heading):.2f}°"
)


# ============================================================
# 19. SANITY CHECK
# ============================================================

print()
print("==============================================")
print("SANITY CHECK")
print("==============================================")


if gps_traveled_distance >= endpoint_distance:

    print(
        "PASS: Traveled distance >= "
        "endpoint displacement"
    )

else:

    print(
        "FAIL: Traveled distance is smaller "
        "than endpoint displacement"
    )


speed_difference = abs(

    gps_traveled_distance
    -
    speed_distance

)


speed_difference_percent = (

    speed_difference
    /
    speed_distance
    *
    100

)


print(
    f"GPS vs speed distance difference: "
    f"{speed_difference:.2f} m"
)

print(
    f"GPS vs speed distance difference: "
    f"{speed_difference_percent:.2f}%"
)


if speed_difference_percent < 5:

    print(
        "PASS: GPS distance and speed "
        "distance are consistent."
    )

elif speed_difference_percent < 10:

    print(
        "WARNING: Moderate difference "
        "between GPS and speed distance."
    )

else:

    print(
        "WARNING: Large difference "
        "between GPS and speed distance."
    )


# ============================================================
# 20. FINAL BENCHMARK VALUES
# ============================================================

print()
print("==============================================")
print("VALUES TO USE FOR DR BENCHMARK")
print("==============================================")


print(
    f"Ground truth East displacement  : "
    f"{east_displacement:.2f} m"
)

print(
    f"Ground truth North displacement : "
    f"{north_displacement:.2f} m"
)

print(
    f"Ground truth endpoint distance  : "
    f"{endpoint_distance:.2f} m"
)

print(
    f"Ground truth traveled distance  : "
    f"{gps_traveled_distance:.2f} m"
)


print()
print("==============================================")
print("GROUND TRUTH VERIFICATION COMPLETE")
print("==============================================")