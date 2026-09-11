from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"


SMARTPHONE_FILE = next(DATA_DIR.rglob("S-S1.csv"))
VEHICLE_FILE = next(DATA_DIR.rglob("V-S1.csv"))


# =========================================================
# LOAD DATA
# =========================================================

print("Loading datasets...")

smart = pd.read_csv(
    SMARTPHONE_FILE,
    encoding="cp1252"
)

vehicle = pd.read_csv(
    VEHICLE_FILE,
    encoding="cp1252"
)


# =========================================================
# COLUMN FINDER
# =========================================================

def find_column(df, text):

    for col in df.columns:

        if text.lower() in str(col).lower():

            return col

    raise ValueError(
        f"Column containing '{text}' not found."
    )


time_col = find_column(
    smart,
    "TIME SINCE START"
)

phone_yaw_col = find_column(
    smart,
    "ORIENTATION (Yaw)"
)

phone_pitch_col = find_column(
    smart,
    "ORIENTATION (Pitch)"
)

phone_roll_col = find_column(
    smart,
    "ORIENTATION (Roll"
)

gps_heading_col = find_column(
    smart,
    "GPS ORIENTATION"
)

vehicle_heading_col = find_column(
    vehicle,
    "Heading"
)

vehicle_speed_col = find_column(
    vehicle,
    "Velocity"
)


# =========================================================
# TIME
# =========================================================

smart["time_seconds"] = (
    pd.to_numeric(
        smart[time_col],
        errors="coerce"
    )
    / 1000.0
)


# =========================================================
# TEST SEGMENT
# =========================================================

START_TIME = 4142.521
END_TIME = 4202.522


mask = (
    (smart["time_seconds"] >= START_TIME)
    &
    (smart["time_seconds"] <= END_TIME)
)


smart_seg = smart.loc[mask].copy()


start_index = smart_seg.index[0]
end_index = smart_seg.index[-1]


vehicle_seg = vehicle.iloc[
    start_index:end_index + 1
].copy()


n = min(
    len(smart_seg),
    len(vehicle_seg)
)


smart_seg = smart_seg.iloc[:n].copy()
vehicle_seg = vehicle_seg.iloc[:n].copy()


print()
print("==============================================")
print("HEADING DIAGNOSTIC")
print("==============================================")

print(
    f"Samples : {n}"
)

print(
    f"Time    : "
    f"{smart_seg['time_seconds'].iloc[0]:.3f}"
    f" -> "
    f"{smart_seg['time_seconds'].iloc[-1]:.3f}"
)


# =========================================================
# EXTRACT HEADINGS
# =========================================================

phone_yaw = (
    pd.to_numeric(
        smart_seg[phone_yaw_col],
        errors="coerce"
    )
    .to_numpy()
)

gps_heading = (
    pd.to_numeric(
        smart_seg[gps_heading_col],
        errors="coerce"
    )
    .to_numpy()
)

vehicle_heading = (
    pd.to_numeric(
        vehicle_seg[vehicle_heading_col],
        errors="coerce"
    )
    .to_numpy()
)

vehicle_speed = (
    pd.to_numeric(
        vehicle_seg[vehicle_speed_col],
        errors="coerce"
    )
    .to_numpy()
)


# =========================================================
# ANGLE HELPERS
# =========================================================

def normalize_angle(angle):

    return (
        angle + 180.0
    ) % 360.0 - 180.0


def circular_difference(a, b):

    return normalize_angle(
        a - b
    )


def circular_mean_deg(values):

    values = np.asarray(values)

    values = values[
        np.isfinite(values)
    ]

    radians = np.radians(values)

    mean_angle = np.arctan2(
        np.mean(np.sin(radians)),
        np.mean(np.cos(radians))
    )

    return (
        np.degrees(mean_angle)
        % 360.0
    )


# =========================================================
# INITIALIZATION
# =========================================================

pre_start = max(
    0,
    start_index - 20
)

pre_phone = (
    smart.iloc[
        pre_start:start_index
    ][phone_yaw_col]
    .astype(float)
    .to_numpy()
)

pre_gps = (
    smart.iloc[
        pre_start:start_index
    ][gps_heading_col]
    .astype(float)
    .to_numpy()
)


phone_initial = circular_mean_deg(
    pre_phone
)

gps_initial = circular_mean_deg(
    pre_gps
)

vehicle_initial = circular_mean_deg(
    vehicle_heading[:10]
)


# =========================================================
# HEADING OFFSET
# =========================================================

heading_offset = normalize_angle(
    gps_initial - phone_initial
)


aligned_phone = (
    phone_yaw
    + heading_offset
) % 360.0


# =========================================================
# ERRORS
# =========================================================

aligned_error = np.array([
    abs(
        circular_difference(
            aligned_phone[i],
            vehicle_heading[i]
        )
    )
    for i in range(n)
])


gps_error = np.array([
    abs(
        circular_difference(
            gps_heading[i],
            vehicle_heading[i]
        )
    )
    for i in range(n)
])


# =========================================================
# OUTPUT
# =========================================================

print()
print("INITIAL HEADINGS")
print("----------------------------------------------")

print(
    f"Phone yaw        : "
    f"{phone_initial:.2f}°"
)

print(
    f"GPS heading      : "
    f"{gps_initial:.2f}°"
)

print(
    f"Vehicle heading  : "
    f"{vehicle_initial:.2f}°"
)

print(
    f"Heading offset   : "
    f"{heading_offset:.2f}°"
)


print()
print("HEADING STATISTICS")
print("----------------------------------------------")

print(
    f"Vehicle heading min : "
    f"{np.nanmin(vehicle_heading):.2f}°"
)

print(
    f"Vehicle heading max : "
    f"{np.nanmax(vehicle_heading):.2f}°"
)

print(
    f"Vehicle heading avg : "
    f"{circular_mean_deg(vehicle_heading):.2f}°"
)


print()
print(
    f"Aligned heading error:"
)

print(
    f"Mean : "
    f"{np.nanmean(aligned_error):.2f}°"
)

print(
    f"Median : "
    f"{np.nanmedian(aligned_error):.2f}°"
)

print(
    f"Maximum : "
    f"{np.nanmax(aligned_error):.2f}°"
)


print()
print(
    f"Raw GPS heading error:"
)

print(
    f"Mean : "
    f"{np.nanmean(gps_error):.2f}°"
)

print(
    f"Median : "
    f"{np.nanmedian(gps_error):.2f}°"
)

print(
    f"Maximum : "
    f"{np.nanmax(gps_error):.2f}°"
)


# =========================================================
# SAMPLE CHECKS
# =========================================================

print()
print("SAMPLE HEADING COMPARISON")
print("----------------------------------------------")

sample_indices = np.linspace(
    0,
    n - 1,
    min(10, n),
    dtype=int
)


for i in sample_indices:

    print(
        f"{smart_seg['time_seconds'].iloc[i]:8.2f}s | "
        f"Phone: {phone_yaw[i]:7.2f}° | "
        f"Aligned: {aligned_phone[i]:7.2f}° | "
        f"Vehicle: {vehicle_heading[i]:7.2f}° | "
        f"Error: {aligned_error[i]:6.2f}°"
    )


# =========================================================
# SPEED
# =========================================================

print()
print("SPEED")
print("----------------------------------------------")

print(
    f"Vehicle average speed : "
    f"{np.nanmean(vehicle_speed):.2f} km/h"
)

print(
    f"Vehicle initial speed : "
    f"{vehicle_speed[0]:.2f} km/h"
)

print(
    f"Vehicle final speed   : "
    f"{vehicle_speed[-1]:.2f} km/h"
)


print()
print("==============================================")
print("HEADING DIAGNOSTIC COMPLETE")
print("==============================================")