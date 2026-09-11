from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"


def find_file(name):
    matches = list(DATA_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found")
    return matches[0]


SMARTPHONE_FILE = find_file("S-S1.csv")


# ============================================================
# LOAD DATA
# ============================================================

print("Loading smartphone dataset...")

smart = pd.read_csv(
    SMARTPHONE_FILE,
    encoding="cp1252"
)


# ============================================================
# FIND COLUMNS
# ============================================================

def find_column(text):
    text = text.upper()

    for col in smart.columns:
        if text in str(col).upper():
            return col

    raise KeyError(f"Column containing '{text}' not found")


time_col = find_column("TIME SINCE START")

acc_x = find_column("ACCELEROMETER X")
acc_y = find_column("ACCELEROMETER Y")
acc_z = find_column("ACCELEROMETER Z")

grav_x = find_column("GRAVITY X")
grav_y = find_column("GRAVITY Y")
grav_z = find_column("GRAVITY Z")

mag_x = find_column("MAGNETIC FIELD X")
mag_y = find_column("MAGNETIC FIELD Y")
mag_z = find_column("MAGNETIC FIELD Z")

gyro_yaw = find_column("GYROSCOPE Yaw")
gyro_pitch = find_column("GYROSCOPE Pitch")
gyro_roll = find_column("GYROSCOPE Roll")

yaw_col = find_column("ORIENTATION (Yaw)")
pitch_col = find_column("ORIENTATION (Pitch)")
roll_col = find_column("ORIENTATION (Roll")


# ============================================================
# TIME
# ============================================================

smart["time_seconds"] = (
    pd.to_numeric(
        smart[time_col],
        errors="coerce"
    ) / 1000.0
)


# ============================================================
# HOLDOUT SEGMENT
# ============================================================

START_TIME = 4142.521
END_TIME = 4202.522

segment = smart[
    (smart["time_seconds"] >= START_TIME) &
    (smart["time_seconds"] <= END_TIME)
].copy()


print()
print("=" * 55)
print("SENSOR VECTOR INSPECTION")
print("=" * 55)

print(f"Samples : {len(segment)}")
print(
    f"Time    : "
    f"{segment['time_seconds'].iloc[0]:.3f} -> "
    f"{segment['time_seconds'].iloc[-1]:.3f}"
)


# ============================================================
# EXTRACT ARRAYS
# ============================================================

acc = segment[
    [acc_x, acc_y, acc_z]
].to_numpy(dtype=float)

gravity = segment[
    [grav_x, grav_y, grav_z]
].to_numpy(dtype=float)

mag = segment[
    [mag_x, mag_y, mag_z]
].to_numpy(dtype=float)

gyro = segment[
    [gyro_yaw, gyro_pitch, gyro_roll]
].to_numpy(dtype=float)


# ============================================================
# MAGNITUDES
# ============================================================

acc_mag = np.linalg.norm(acc, axis=1)
gravity_mag = np.linalg.norm(gravity, axis=1)
mag_mag = np.linalg.norm(mag, axis=1)
gyro_mag = np.linalg.norm(gyro, axis=1)


# ============================================================
# PRINT STATISTICS
# ============================================================

print()
print("ACCELEROMETER")
print("-" * 55)

print(
    f"X : mean={np.mean(acc[:,0]):.4f} "
    f"min={np.min(acc[:,0]):.4f} "
    f"max={np.max(acc[:,0]):.4f}"
)

print(
    f"Y : mean={np.mean(acc[:,1]):.4f} "
    f"min={np.min(acc[:,1]):.4f} "
    f"max={np.max(acc[:,1]):.4f}"
)

print(
    f"Z : mean={np.mean(acc[:,2]):.4f} "
    f"min={np.min(acc[:,2]):.4f} "
    f"max={np.max(acc[:,2]):.4f}"
)

print(
    f"Magnitude : mean={np.mean(acc_mag):.4f} "
    f"min={np.min(acc_mag):.4f} "
    f"max={np.max(acc_mag):.4f}"
)


print()
print("GRAVITY")
print("-" * 55)

print(
    f"X : mean={np.mean(gravity[:,0]):.4f} "
    f"min={np.min(gravity[:,0]):.4f} "
    f"max={np.max(gravity[:,0]):.4f}"
)

print(
    f"Y : mean={np.mean(gravity[:,1]):.4f} "
    f"min={np.min(gravity[:,1]):.4f} "
    f"max={np.max(gravity[:,1]):.4f}"
)

print(
    f"Z : mean={np.mean(gravity[:,2]):.4f} "
    f"min={np.min(gravity[:,2]):.4f} "
    f"max={np.max(gravity[:,2]):.4f}"
)

print(
    f"Magnitude : mean={np.mean(gravity_mag):.4f} "
    f"min={np.min(gravity_mag):.4f} "
    f"max={np.max(gravity_mag):.4f}"
)


print()
print("MAGNETOMETER")
print("-" * 55)

print(
    f"X : mean={np.mean(mag[:,0]):.4f} "
    f"min={np.min(mag[:,0]):.4f} "
    f"max={np.max(mag[:,0]):.4f}"
)

print(
    f"Y : mean={np.mean(mag[:,1]):.4f} "
    f"min={np.min(mag[:,1]):.4f} "
    f"max={np.max(mag[:,1]):.4f}"
)

print(
    f"Z : mean={np.mean(mag[:,2]):.4f} "
    f"min={np.min(mag[:,2]):.4f} "
    f"max={np.max(mag[:,2]):.4f}"
)

print(
    f"Magnitude : mean={np.mean(mag_mag):.4f} "
    f"min={np.min(mag_mag):.4f} "
    f"max={np.max(mag_mag):.4f}"
)


print()
print("GYROSCOPE")
print("-" * 55)

print(
    f"Yaw : mean={np.mean(gyro[:,0]):.4f} "
    f"min={np.min(gyro[:,0]):.4f} "
    f"max={np.max(gyro[:,0]):.4f}"
)

print(
    f"Pitch : mean={np.mean(gyro[:,1]):.4f} "
    f"min={np.min(gyro[:,1]):.4f} "
    f"max={np.max(gyro[:,1]):.4f}"
)

print(
    f"Roll : mean={np.mean(gyro[:,2]):.4f} "
    f"min={np.min(gyro[:,2]):.4f} "
    f"max={np.max(gyro[:,2]):.4f}"
)

print(
    f"Magnitude : mean={np.mean(gyro_mag):.4f} "
    f"min={np.min(gyro_mag):.4f} "
    f"max={np.max(gyro_mag):.4f}"
)


# ============================================================
# ORIENTATION
# ============================================================

print()
print("ANDROID ORIENTATION")
print("-" * 55)

for name, col in [
    ("Yaw", yaw_col),
    ("Pitch", pitch_col),
    ("Roll", roll_col)
]:

    values = pd.to_numeric(
        segment[col],
        errors="coerce"
    ).to_numpy()

    print(
        f"{name:<7}: "
        f"mean={np.nanmean(values):.2f}° "
        f"min={np.nanmin(values):.2f}° "
        f"max={np.nanmax(values):.2f}°"
    )


# ============================================================
# FIRST / MIDDLE / LAST SENSOR VECTORS
# ============================================================

print()
print("SAMPLE SENSOR VECTORS")
print("-" * 55)

indices = [
    0,
    len(segment) // 2,
    len(segment) - 1
]

for i in indices:

    row = segment.iloc[i]

    print()
    print(
        f"Time: {row['time_seconds']:.3f}s"
    )

    print(
        "Accel   : "
        f"[{row[acc_x]:.4f}, "
        f"{row[acc_y]:.4f}, "
        f"{row[acc_z]:.4f}]"
    )

    print(
        "Gravity : "
        f"[{row[grav_x]:.4f}, "
        f"{row[grav_y]:.4f}, "
        f"{row[grav_z]:.4f}]"
    )

    print(
        "Magnetic: "
        f"[{row[mag_x]:.4f}, "
        f"{row[mag_y]:.4f}, "
        f"{row[mag_z]:.4f}]"
    )

    print(
        "Gyro    : "
        f"[{row[gyro_yaw]:.4f}, "
        f"{row[gyro_pitch]:.4f}, "
        f"{row[gyro_roll]:.4f}]"
    )


print()
print("=" * 55)
print("SENSOR VECTOR INSPECTION COMPLETE")
print("=" * 55)