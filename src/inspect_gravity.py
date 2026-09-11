import pandas as pd
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

gravity_x = find_column("GRAVITY X")
gravity_y = find_column("GRAVITY Y")
gravity_z = find_column("GRAVITY Z")

accel_x = find_column("ACCELEROMETER X")
accel_y = find_column("ACCELEROMETER Y")
accel_z = find_column("ACCELEROMETER Z")

time = pd.to_numeric(df[time_column], errors="coerce") / 1000

gx = pd.to_numeric(df[gravity_x], errors="coerce")
gy = pd.to_numeric(df[gravity_y], errors="coerce")
gz = pd.to_numeric(df[gravity_z], errors="coerce")

ax = pd.to_numeric(df[accel_x], errors="coerce")
ay = pd.to_numeric(df[accel_y], errors="coerce")
az = pd.to_numeric(df[accel_z], errors="coerce")

print("======================================")
print("       GRAVITY INSPECTION")
print("======================================")
print()

print("Gravity columns:")
print(gravity_x)
print(gravity_y)
print(gravity_z)

print()

print("Average gravity:")
print("X:", gx.mean())
print("Y:", gy.mean())
print("Z:", gz.mean())

print()

print("Gravity magnitude:")

gravity_magnitude = (
    gx**2 +
    gy**2 +
    gz**2
) ** 0.5

print(
    "Average:",
    gravity_magnitude.mean()
)

print(
    "Minimum:",
    gravity_magnitude.min()
)

print(
    "Maximum:",
    gravity_magnitude.max()
)

print()

print("======================================")

plt.figure(figsize=(10, 6))

plt.plot(time, gx, label="Gravity X")
plt.plot(time, gy, label="Gravity Y")
plt.plot(time, gz, label="Gravity Z")

plt.xlabel("Time (seconds)")
plt.ylabel("Gravity (m/s²)")
plt.title("S-S1 Gravity Measurements")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

print()
print("Gravity inspection completed.")