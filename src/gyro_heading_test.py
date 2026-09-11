from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"


def find_file(name):
    matches = list(DATA_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found")
    return matches[0]


SMARTPHONE_FILE = find_file("S-S1.csv")
VEHICLE_FILE = find_file("V-S1.csv")


# ============================================================
# LOAD
# ============================================================

print("Loading datasets...")

smart = pd.read_csv(
    SMARTPHONE_FILE,
    encoding="cp1252"
)

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
        f"Could not find column containing: {text}"
    )


smart_time = find_column(
    smart,
    "TIME SINCE START"
)

gyro_z = find_column(
    smart,
    "GYROSCOPE Yaw"
)

vehicle_heading = find_column(
    vehicle,
    "Heading"
)


# ============================================================
# TIME
# ============================================================

smart["time_seconds"] = (
    pd.to_numeric(
        smart[smart_time],
        errors="coerce"
    ) / 1000.0
)


# ============================================================
# SEGMENT
# ============================================================

START_TIME = 4142.521
END_TIME = 4202.522

mask = (
    (smart["time_seconds"] >= START_TIME) &
    (smart["time_seconds"] <= END_TIME)
)

segment = smart[mask].copy()

start_index = segment.index[0]
end_index = segment.index[-1]


# ============================================================
# MATCH VEHICLE ROWS
# ============================================================

vehicle_segment = vehicle.iloc[
    start_index:end_index + 1
].copy()


# ============================================================
# ARRAYS
# ============================================================

t = segment["time_seconds"].to_numpy()

gyro = pd.to_numeric(
    segment[gyro_z],
    errors="coerce"
).to_numpy()

heading_gt = pd.to_numeric(
    vehicle_segment[vehicle_heading],
    errors="coerce"
).to_numpy()


# ============================================================
# INTEGRATE GYRO
# ============================================================

gyro_heading = np.zeros(len(gyro))

for i in range(1, len(gyro)):

    dt = t[i] - t[i - 1]

    gyro_heading[i] = (
        gyro_heading[i - 1]
        + np.degrees(
            gyro[i] * dt
        )
    )


# ============================================================
# GROUND TRUTH HEADING CHANGE
# ============================================================

def angle_diff(a, b):

    return (
        (a - b + 180) % 360
    ) - 180


gt_change = angle_diff(
    heading_gt[-1],
    heading_gt[0]
)

gyro_change = gyro_heading[-1]


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 55)
print("GYROSCOPE HEADING TEST")
print("=" * 55)

print(
    f"Samples : {len(segment)}"
)

print(
    f"Time    : {t[0]:.3f} -> {t[-1]:.3f}"
)

print()

print("GROUND TRUTH")
print("-" * 55)

print(
    f"Initial vehicle heading : "
    f"{heading_gt[0]:.2f}°"
)

print(
    f"Final vehicle heading   : "
    f"{heading_gt[-1]:.2f}°"
)

print(
    f"Vehicle heading change  : "
    f"{gt_change:.2f}°"
)

print()

print("GYROSCOPE")
print("-" * 55)

print(
    f"Integrated gyro change  : "
    f"{gyro_change:.2f}°"
)

print(
    f"Gyro initial heading    : "
    f"0.00°"
)

print(
    f"Gyro final relative heading : "
    f"{gyro_heading[-1]:.2f}°"
)

print()

print("ERROR")
print("-" * 55)

print(
    f"Heading-change error    : "
    f"{gyro_change - gt_change:.2f}°"
)

print(
    f"Absolute error          : "
    f"{abs(gyro_change - gt_change):.2f}°"
)

print()

print("=" * 55)
print("GYROSCOPE HEADING TEST COMPLETE")
print("=" * 55)