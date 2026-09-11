from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"


def find_file(name):

    matches = list(DATA_DIR.rglob(name))

    if not matches:
        raise FileNotFoundError(
            f"{name} not found inside {DATA_DIR}"
        )

    return matches[0]


VEHICLE_FILE = find_file("V-S1.csv")


# ============================================================
# LOAD
# ============================================================

vehicle = pd.read_csv(
    VEHICLE_FILE,
    encoding="cp1252"
)


# ============================================================
# FIND COLUMNS
# ============================================================

def find_column(df, text):

    text = text.upper()

    for col in df.columns:

        if text in str(col).upper():
            return col

    raise KeyError(
        f"Column containing '{text}' not found"
    )


time_col = find_column(
    vehicle,
    "Time"
)

heading_col = find_column(
    vehicle,
    "Heading"
)


# ============================================================
# DATA
# ============================================================

time = pd.to_numeric(
    vehicle[time_col],
    errors="coerce"
).to_numpy()


heading = pd.to_numeric(
    vehicle[heading_col],
    errors="coerce"
).to_numpy()


# ============================================================
# UNWRAP
# ============================================================

heading_unwrapped = np.rad2deg(
    np.unwrap(
        np.deg2rad(heading)
    )
)


# ============================================================
# YAW RATE
# ============================================================

dt = np.diff(time)

heading_change = np.diff(
    heading_unwrapped
)

yaw_rate = (
    heading_change / dt
)


# ============================================================
# BASIC STATISTICS
# ============================================================

valid = (
    np.isfinite(yaw_rate)
    &
    np.isfinite(dt)
    &
    (dt > 0.05)
    &
    (dt < 0.2)
)


yaw_rate_valid = yaw_rate[valid]


print()
print("=" * 60)
print("VEHICLE YAW-RATE DIAGNOSTIC")
print("=" * 60)

print(
    f"Total samples : {len(vehicle)}"
)

print(
    f"Valid samples : {len(yaw_rate_valid)}"
)

print()
print("RAW YAW-RATE")

print(
    f"Minimum : {np.min(yaw_rate_valid):.3f} deg/s"
)

print(
    f"Maximum : {np.max(yaw_rate_valid):.3f} deg/s"
)

print(
    f"Mean    : {np.mean(yaw_rate_valid):.3f} deg/s"
)

print(
    f"Median  : {np.median(yaw_rate_valid):.3f} deg/s"
)

print(
    f"Std     : {np.std(yaw_rate_valid):.3f} deg/s"
)


# ============================================================
# PERCENTILES
# ============================================================

print()
print("PERCENTILES")

for p in [
    1,
    5,
    25,
    50,
    75,
    95,
    99
]:

    print(
        f"{p:>2}% : "
        f"{np.percentile(yaw_rate_valid, p):.3f} deg/s"
    )


# ============================================================
# EXTREME VALUES
# ============================================================

print()
print("=" * 60)
print("LARGEST YAW-RATE VALUES")
print("=" * 60)


indices = np.argsort(
    np.abs(yaw_rate_valid)
)[-20:][::-1]


valid_indices = np.where(valid)[0]


for rank, idx in enumerate(
    indices,
    start=1
):

    original_idx = (
        valid_indices[idx]
    )

    print(
        f"{rank:02d}. "
        f"row={original_idx + 1} "
        f"time={time[original_idx + 1]:.3f}s "
        f"heading_before={heading[original_idx]:.3f}° "
        f"heading_after={heading[original_idx + 1]:.3f}° "
        f"change="
        f"{heading_unwrapped[original_idx + 1] - heading_unwrapped[original_idx]:.3f}° "
        f"dt={dt[original_idx]:.3f}s "
        f"yaw_rate={yaw_rate[original_idx]:.3f}°/s"
    )


# ============================================================
# REALISTIC RANGE
# ============================================================

reasonable = (
    np.abs(yaw_rate_valid) < 20
)


print()
print("=" * 60)
print("REASONABLE YAW-RATE RANGE")
print("=" * 60)

print(
    f"Samples below ±20 deg/s : "
    f"{np.sum(reasonable)}"
)

print(
    f"Percentage               : "
    f"{100 * np.mean(reasonable):.2f}%"
)


# ============================================================
# OUTLIERS
# ============================================================

print()

outliers = (
    np.abs(yaw_rate_valid) >= 20
)

print(
    f"Samples >= ±20 deg/s : "
    f"{np.sum(outliers)}"
)

print(
    f"Percentage           : "
    f"{100 * np.mean(outliers):.2f}%"
)

print()
print("Diagnostic complete.")