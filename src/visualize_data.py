import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Find project folder
project_folder = Path(__file__).resolve().parent.parent

# Find dataset
data_folder = project_folder / "Data"
files = list(data_folder.rglob("S-S1.csv"))

if not files:
    raise FileNotFoundError("S-S1.csv was not found inside the Data folder.")

file_path = files[0]

print("Dataset found:")
print(file_path)
print()


# Load dataset
df = pd.read_csv(file_path, encoding="cp1252")

# Clean column names
df.columns = df.columns.str.strip()

print("Dataset loaded successfully")
print("Shape:", df.shape)
print()


# Function to find a column safely
def find_column(keyword):
    for column in df.columns:
        if keyword.lower() in column.lower():
            return column

    return None


# Find important columns
time_column = find_column("TIME SINCE START")
latitude_column = find_column("GPS LATITUDE")
longitude_column = find_column("GPS LONGITUDE")
speed_column = find_column("GPS SPEED")


# Check time column
if time_column is None:
    raise KeyError("TIME SINCE START column not found.")

# Convert milliseconds to seconds
time = df[time_column] / 1000


print("Columns detected:")
print("Time      :", time_column)
print("Latitude  :", latitude_column)
print("Longitude :", longitude_column)
print("Speed     :", speed_column)
print()


# --------------------------------------------------
# 1. GPS TRAJECTORY
# --------------------------------------------------

if latitude_column is not None and longitude_column is not None:

    plt.figure(figsize=(10, 6))

    plt.plot(
        df[longitude_column],
        df[latitude_column]
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("S-S1 GPS Trajectory")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 2. ACCELEROMETER DATA
# --------------------------------------------------

accelerometer_x = find_column("ACCELEROMETER X")
accelerometer_y = find_column("ACCELEROMETER Y")
accelerometer_z = find_column("ACCELEROMETER Z")


if (
    accelerometer_x is not None
    and accelerometer_y is not None
    and accelerometer_z is not None
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        time,
        df[accelerometer_x],
        label="Accelerometer X"
    )

    plt.plot(
        time,
        df[accelerometer_y],
        label="Accelerometer Y"
    )

    plt.plot(
        time,
        df[accelerometer_z],
        label="Accelerometer Z"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("S-S1 Accelerometer Data")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 3. GYROSCOPE DATA
# --------------------------------------------------

gyro_yaw = find_column("GYROSCOPE Yaw")
gyro_pitch = find_column("GYROSCOPE Pitch")
gyro_roll = find_column("GYROSCOPE Roll")


if (
    gyro_yaw is not None
    and gyro_pitch is not None
    and gyro_roll is not None
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        time,
        df[gyro_yaw],
        label="Gyroscope Yaw"
    )

    plt.plot(
        time,
        df[gyro_pitch],
        label="Gyroscope Pitch"
    )

    plt.plot(
        time,
        df[gyro_roll],
        label="Gyroscope Roll"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.title("S-S1 Gyroscope Data")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 4. GPS SPEED
# --------------------------------------------------

if speed_column is not None:

    plt.figure(figsize=(10, 6))

    plt.plot(
        time,
        df[speed_column]
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Speed (km/h)")
    plt.title("S-S1 GPS Speed")

    plt.grid(True)

    plt.tight_layout()
    plt.show()

else:

    print("GPS SPEED column was not detected.")
    print()
    print("Available columns containing GPS:")
    
    for column in df.columns:
        if "GPS" in column.upper():
            print(repr(column))


print()
print("Visualization completed successfully.")