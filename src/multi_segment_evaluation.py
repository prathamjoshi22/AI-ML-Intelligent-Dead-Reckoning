import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "Data"

SMARTPHONE_FILE = next(DATA_DIR.rglob("S-S1.csv"))
VEHICLE_FILE = next(DATA_DIR.rglob("V-S1.csv"))

FUSION_SCRIPT = SRC_DIR / "ai_fusion_dead_reckoning.py"
FUSION_RESULT = BASE_DIR / "ai_fusion_results.csv"

RESULT_DIR = BASE_DIR / "multi_segment_results"
RESULT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading datasets...")

smart = pd.read_csv(SMARTPHONE_FILE, encoding="cp1252")
vehicle = pd.read_csv(VEHICLE_FILE, encoding="cp1252")

time_col = None

for col in smart.columns:
    if "TIME SINCE START" in str(col).upper():
        time_col = col
        break

if time_col is None:
    print("ERROR: TIME SINCE START column not found.")
    print("Available columns:")
    for col in smart.columns:
        print(repr(col))
    sys.exit(1)

print(f"Using smartphone time column: {time_col}")

smart["time_seconds"] = (
    pd.to_numeric(smart[time_col], errors="coerce")
    / 1000.0
)

print(f"Smartphone rows: {len(smart)}")
print(f"Vehicle rows   : {len(vehicle)}")


# ---------------------------------------------------------
# PROPER HOLDOUT REGION
# ---------------------------------------------------------

split_index = int(len(smart) * 0.80)

holdout_start = smart["time_seconds"].iloc[split_index]

holdout_end = smart["time_seconds"].iloc[-1] - 60.0

print()
print("==============================================")
print("PROPER UNSEEN-DATA EVALUATION")
print("==============================================")

print(f"Training region : 0 -> {holdout_start:.2f}s")
print(f"Holdout region  : {holdout_start:.2f}s -> {holdout_end:.2f}s")

print()
print("The AI model was trained on the first 80%.")
print("Only the final 20% will be used for testing.")


# ---------------------------------------------------------
# VEHICLE COLUMN FINDER
# ---------------------------------------------------------

def find_column(df, text):
    for col in df.columns:
        if text.lower() in str(col).lower():
            return col

    raise ValueError(f"Column containing '{text}' not found.")


vehicle_lat_col = find_column(vehicle, "Latitude")
vehicle_lon_col = find_column(vehicle, "Longitude")
vehicle_speed_col = find_column(vehicle, "Velocity")


# ---------------------------------------------------------
# HAVERSINE
# ---------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):

    R = 6371000.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------
# LOCAL COORDINATES
# ---------------------------------------------------------

def local_xy(lat, lon, lat0, lon0):

    R = 6371000.0

    lat = np.radians(lat)
    lon = np.radians(lon)

    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    north = (lat - lat0) * R

    east = (
        (lon - lon0)
        * R
        * np.cos(lat0)
    )

    return east, north


# ---------------------------------------------------------
# FIND VALID HOLDOUT SEGMENTS
# ---------------------------------------------------------

print()
print("Searching for valid 60-second segments in holdout region...")

candidates = []

current_time = holdout_start

while current_time <= holdout_end:

    end_time = current_time + 60.0

    start_idx = (
        smart["time_seconds"]
        .sub(
            current_time
        )
        .abs()
        .idxmin()
    )

    end_idx = (
        smart["time_seconds"]
        .sub(
            end_time
        )
        .abs()
        .idxmin()
    )

    if end_idx <= start_idx:
        current_time += 300.0
        continue

    vehicle_segment = vehicle.iloc[
        start_idx:end_idx + 1
    ].copy()

    if len(vehicle_segment) < 500:
        current_time += 300.0
        continue

    speed = vehicle_segment[
        vehicle_speed_col
    ].astype(float)

    avg_speed = speed.mean()

    lat = vehicle_segment[
        vehicle_lat_col
    ].astype(float).to_numpy()

    lon = vehicle_segment[
        vehicle_lon_col
    ].astype(float).to_numpy()

    distance = np.sum(
        haversine(
            lat[:-1],
            lon[:-1],
            lat[1:],
            lon[1:]
        )
    )

    if avg_speed >= 20.0 and distance >= 400.0:

        candidates.append({
            "start": smart["time_seconds"].iloc[start_idx],
            "end": smart["time_seconds"].iloc[end_idx],
            "speed": avg_speed,
            "distance": distance,
            "start_idx": start_idx,
            "end_idx": end_idx
        })

    current_time += 300.0


print(f"Candidate segments found: {len(candidates)}")


# ---------------------------------------------------------
# SELECT UP TO 5
# ---------------------------------------------------------

selected = []

for candidate in candidates:

    if len(selected) == 5:
        break

    if not selected:

        selected.append(candidate)
        continue

    previous = selected[-1]

    if (
        candidate["start"]
        - previous["start"]
        >= 300
    ):

        selected.append(candidate)


print(f"Selected segments: {len(selected)}")


if not selected:

    print()
    print("ERROR: No valid holdout segments found.")
    sys.exit(1)


print()
print("==============================================")
print("SELECTED UNSEEN TEST SEGMENTS")
print("==============================================")

for i, seg in enumerate(selected, 1):

    print(
        f"Segment {i}: "
        f"{seg['start']:.3f}s -> "
        f"{seg['end']:.3f}s | "
        f"Speed: {seg['speed']:.2f} km/h | "
        f"Distance: {seg['distance']:.2f} m"
    )


# ---------------------------------------------------------
# RUN FUSION
# ---------------------------------------------------------

summary = []


for number, seg in enumerate(selected, 1):

    start = seg["start"]
    end = seg["end"]

    print()
    print("==================================================")
    print(f"RUNNING UNSEEN SEGMENT {number}")
    print(f"{start:.3f}s -> {end:.3f}s")
    print("==================================================")

    env = os.environ.copy()

    env["GNSS_START_TIME"] = str(start)
    env["GNSS_END_TIME"] = str(end)

    result = subprocess.run(
        [
            sys.executable,
            str(FUSION_SCRIPT)
        ],
        cwd=str(BASE_DIR),
        env=env
    )

    if result.returncode != 0:

        print(
            f"Fusion failed for segment {number}"
        )

        continue


    # -----------------------------------------------------
    # LOAD DR RESULT
    # -----------------------------------------------------

    if not FUSION_RESULT.exists():

        print(
            "ERROR: ai_fusion_results.csv "
            "was not created."
        )

        continue


    dr = pd.read_csv(FUSION_RESULT)

    if len(dr) == 0:

        print(
            f"No DR data for segment {number}"
        )

        continue


    # -----------------------------------------------------
    # MATCH VEHICLE DATA
    # -----------------------------------------------------

    start_idx = seg["start_idx"]
    end_idx = seg["end_idx"]

    vehicle_segment = vehicle.iloc[
        start_idx:end_idx + 1
    ].copy()

    n = min(
        len(dr),
        len(vehicle_segment)
    )

    dr = dr.iloc[:n].copy()

    vehicle_segment = vehicle_segment.iloc[
        :n
    ].copy()


    # -----------------------------------------------------
    # GROUND TRUTH
    # -----------------------------------------------------

    gt_lat = vehicle_segment[
        vehicle_lat_col
    ].astype(float).to_numpy()

    gt_lon = vehicle_segment[
        vehicle_lon_col
    ].astype(float).to_numpy()


    # -----------------------------------------------------
    # DR POSITION
    # -----------------------------------------------------

    dr_lat = dr["latitude"].astype(
        float
    ).to_numpy()

    dr_lon = dr["longitude"].astype(
        float
    ).to_numpy()


    # -----------------------------------------------------
    # LOCAL COORDINATES
    # -----------------------------------------------------

    gt_east, gt_north = local_xy(
        gt_lat,
        gt_lon,
        gt_lat[0],
        gt_lon[0]
    )

    dr_east, dr_north = local_xy(
        dr_lat,
        dr_lon,
        dr_lat[0],
        dr_lon[0]
    )


    # -----------------------------------------------------
    # POSITION ERROR
    # -----------------------------------------------------

    errors = np.sqrt(
        (dr_east - gt_east) ** 2
        +
        (dr_north - gt_north) ** 2
    )

    final_error = errors[-1]

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    average_error = np.mean(errors)

    maximum_error = np.max(errors)


    # -----------------------------------------------------
    # GROUND TRUTH DISTANCE
    # -----------------------------------------------------

    gt_distance = np.sum(
        haversine(
            gt_lat[:-1],
            gt_lon[:-1],
            gt_lat[1:],
            gt_lon[1:]
        )
    )


    # -----------------------------------------------------
    # DRIFT
    # -----------------------------------------------------

    drift = (
        final_error
        /
        gt_distance
        *
        100
    )


    # -----------------------------------------------------
    # SAVE INDIVIDUAL RESULT
    # -----------------------------------------------------

    output = pd.DataFrame({

        "time_seconds":
            dr["time_seconds"],

        "ground_truth_latitude":
            gt_lat,

        "ground_truth_longitude":
            gt_lon,

        "dr_latitude":
            dr_lat,

        "dr_longitude":
            dr_lon,

        "position_error_m":
            errors

    })


    output_file = (
        RESULT_DIR
        /
        f"holdout_segment_{number}.csv"
    )

    output.to_csv(
        output_file,
        index=False
    )


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    summary.append({

        "segment":
            number,

        "start_time_s":
            start,

        "end_time_s":
            end,

        "ground_truth_distance_m":
            gt_distance,

        "final_error_m":
            final_error,

        "drift_percent":
            drift,

        "trajectory_rmse_m":
            rmse,

        "average_error_m":
            average_error,

        "maximum_error_m":
            maximum_error

    })


    print()
    print(
        f"UNSEEN SEGMENT {number} RESULT"
    )

    print(
        f"Ground truth distance : "
        f"{gt_distance:.2f} m"
    )

    print(
        f"Final position error  : "
        f"{final_error:.2f} m"
    )

    print(
        f"Drift                 : "
        f"{drift:.2f}%"
    )

    print(
        f"Trajectory RMSE       : "
        f"{rmse:.2f} m"
    )


# ---------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------

summary_df = pd.DataFrame(summary)

summary_file = (
    RESULT_DIR
    /
    "holdout_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


print()
print("==================================================")
print("UNSEEN-DATA FINAL REPORT")
print("==================================================")

if len(summary_df) > 0:

    print(
        summary_df[
            [
                "segment",
                "start_time_s",
                "end_time_s",
                "ground_truth_distance_m",
                "final_error_m",
                "drift_percent",
                "trajectory_rmse_m"
            ]
        ].to_string(index=False)
    )


    average_drift = (
        summary_df[
            "drift_percent"
        ].mean()
    )

    best_drift = (
        summary_df[
            "drift_percent"
        ].min()
    )

    worst_drift = (
        summary_df[
            "drift_percent"
        ].max()
    )

    average_error = (
        summary_df[
            "final_error_m"
        ].mean()
    )


    print()
    print(
        f"Segments tested : "
        f"{len(summary_df)}"
    )

    print(
        f"Average drift   : "
        f"{average_drift:.2f}%"
    )

    print(
        f"Best drift      : "
        f"{best_drift:.2f}%"
    )

    print(
        f"Worst drift     : "
        f"{worst_drift:.2f}%"
    )

    print(
        f"Average error   : "
        f"{average_error:.2f} m"
    )


    print()
    print("==================================================")
    print("SIH <10% DRIFT CHECK")
    print("==================================================")

    if average_drift < 10:

        print(
            "PASS: Average unseen-data "
            "drift is below 10%."
        )

    else:

        print(
            "CURRENT SYSTEM DOES NOT "
            "MEET <10% AVERAGE DRIFT."
        )

else:

    print(
        "No segments were successfully evaluated."
    )


print()
print(
    f"Summary saved to:\n{summary_file}"
)

print()
print(
    "=================================================="
)

print(
    "UNSEEN-DATA EVALUATION COMPLETE"
)

print(
    "=================================================="
)