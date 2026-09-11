import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "Data"

files = list(data_folder.rglob("S-S1.csv"))

if not files:
    raise FileNotFoundError("S-S1.csv was not found.")

file_path = files[0]

df = pd.read_csv(file_path, encoding="cp1252")
df.columns = df.columns.str.strip()

def find_column(keyword):
    for column in df.columns:
        if keyword.lower() in column.lower():
            return column
    return None

time_column = find_column("TIME SINCE START")

yaw_column = find_column("ORIENTATION (Yaw)")
pitch_column = find_column("ORIENTATION (Pitch)")
roll_column = find_column("ORIENTATION (Roll")

gps_orientation_column = find_column("GPS ORIENTATION")
speed_column = find_column("GPS SPEED")

time = pd.to_numeric(
    df[time_column],
    errors="coerce"
) / 1000

yaw = pd.to_numeric(
    df[yaw_column],
    errors="coerce"
)

pitch = pd.to_numeric(
    df[pitch_column],
    errors="coerce"
)

roll = pd.to_numeric(
    df[roll_column],
    errors="coerce"
)

gps_orientation = pd.to_numeric(
    df[gps_orientation_column],
    errors="coerce"
)

speed = pd.to_numeric(
    df[speed_column],
    errors="coerce"
)

start_time = 3562
end_time = 3622

mask = (
    (time >= start_time) &
    (time <= end_time)
)

t = time[mask].to_numpy()
y = yaw[mask].to_numpy()
p = pitch[mask].to_numpy()
r = roll[mask].to_numpy()
go = gps_orientation[mask].to_numpy()
s = speed[mask].to_numpy()

print("======================================")
print("       ORIENTATION INSPECTION")
print("======================================")
print()

print("Interval:")
print(f"{start_time}s → {end_time}s")

print()

print("YAW")
print("Minimum:", y.min())
print("Maximum:", y.max())
print("Average:", y.mean())

print()

print("PITCH")
print("Minimum:", p.min())
print("Maximum:", p.max())
print("Average:", p.mean())

print()

print("ROLL")
print("Minimum:", r.min())
print("Maximum:", r.max())
print("Average:", r.mean())

print()

print("GPS ORIENTATION")
print("Minimum:", go.min())
print("Maximum:", go.max())
print("Average:", go.mean())

print()

print("GPS SPEED")
print("Minimum:", s.min())
print("Maximum:", s.max())
print("Average:", s.mean())

print()

print("======================================")

plt.figure(figsize=(10, 6))

plt.plot(t, y, label="Orientation Yaw")
plt.plot(t, p, label="Orientation Pitch")
plt.plot(t, r, label="Orientation Roll")

plt.xlabel("Time (seconds)")
plt.ylabel("Angle (degrees)")
plt.title("Phone Orientation During GNSS Blackout")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

print()
print("Orientation inspection completed.")