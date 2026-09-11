import pandas as pd
import numpy as np
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "Data"

files = list(data_folder.rglob("V-S1.csv"))

if not files:
    raise FileNotFoundError("V-S1.csv was not found.")

file_path = files[0]

df = pd.read_csv(file_path, encoding="cp1252")
df.columns = df.columns.str.strip()

time_col = "Time Since Start of Day (seconds)"
lat_col = "Latitude (degrees)"
lon_col = "Longitude (degrees)"
speed_col = "Velocity (km/hr)"
heading_col = "Heading (degrees)"
sample_col = "Sample period (seconds)"
yaw_col = "Yaw Rate (deg/sec)"
long_acc_col = "Indicated Longitudinal Acceleration (g)"
lat_acc_col = "Indicated Lateral Acceleration (g)"

print("==============================================")
print("       VEHICLE REFERENCE VALIDATION")
print("==============================================")
print()

print("Vehicle time range:")
print("Start:", df[time_col].iloc[0])
print("End  :", df[time_col].iloc[-1])
print()

print("Vehicle duration:")
print(df[time_col].iloc[-1] - df[time_col].iloc[0], "seconds")
print()

# Smartphone relative interval
smartphone_start = 3562
smartphone_end = 3622

# Estimate offset using the beginning of both synchronized datasets
vehicle_start = df[time_col].iloc[0]

vehicle_start_time = vehicle_start + smartphone_start
vehicle_end_time = vehicle_start + smartphone_end

print("Requested smartphone interval:")
print(f"{smartphone_start}s → {smartphone_end}s")
print()

print("Corresponding vehicle interval:")
print(f"{vehicle_start_time}s → {vehicle_end_time}s")
print()

segment = df[
    (df[time_col] >= vehicle_start_time) &
    (df[time_col] <= vehicle_end_time)
].copy()

if segment.empty:
    raise ValueError(
        "No vehicle data found in the selected interval."
    )

print("Number of samples:", len(segment))
print()

print("==============================================")
print("                 TIME")
print("==============================================")
print()

first_time = segment[time_col].iloc[0]
last_time = segment[time_col].iloc[-1]

print("First time:", first_time)
print("Last time :", last_time)
print("Duration  :", last_time - first_time)
print()

print("==============================================")
print("                VELOCITY")
print("==============================================")
print()

speed = segment[speed_col]

print("Minimum:", speed.min(), "km/h")
print("Maximum:", speed.max(), "km/h")
print("Average:", speed.mean(), "km/h")
print()

print("==============================================")
print("              GPS POSITION")
print("==============================================")
print()

lat1 = segment[lat_col].iloc[0]
lon1 = segment[lon_col].iloc[0]

lat2 = segment[lat_col].iloc[-1]
lon2 = segment[lon_col].iloc[-1]

print("Starting latitude :", lat1)
print("Starting longitude:", lon1)

print("Ending latitude   :", lat2)
print("Ending longitude  :", lon2)
print()

lat_avg = np.radians((lat1 + lat2) / 2)

north = (lat2 - lat1) * 111320
east = (lon2 - lon1) * 111320 * np.cos(lat_avg)

position_distance = np.sqrt(
    north**2 + east**2
)

print("North displacement:", north, "meters")
print("East displacement :", east, "meters")
print("Straight-line displacement:",
      position_distance, "meters")
print()

print("==============================================")
print("          GPS TRAVELED DISTANCE")
print("==============================================")
print()

lat = np.radians(segment[lat_col].values)
lon = np.radians(segment[lon_col].values)

dlat = np.diff(lat)
dlon = np.diff(lon)

lat_mid = (lat[:-1] + lat[1:]) / 2

north_distance = dlat * 6371000
east_distance = dlon * 6371000 * np.cos(lat_mid)

step_distance = np.sqrt(
    north_distance**2 +
    east_distance**2
)

gps_distance = np.sum(step_distance)

print("Total GPS traveled distance:",
      gps_distance, "meters")
print()

print("==============================================")
print("          SPEED BASED DISTANCE")
print("==============================================")
print()

time_seconds = segment[time_col].values

dt = np.diff(time_seconds)

speed_ms = speed.values[:-1] / 3.6

speed_distance = np.sum(speed_ms * dt)

print("Distance from vehicle speed:",
      speed_distance, "meters")
print()

average_speed_distance = (
    speed.mean() / 3.6 *
    (time_seconds[-1] - time_seconds[0])
)

print("Average speed × duration:",
      average_speed_distance, "meters")
print()

print("==============================================")
print("                HEADING")
print("==============================================")
print()

heading = segment[heading_col]

print("Minimum:", heading.min())
print("Maximum:", heading.max())
print("Average:", heading.mean())
print()

print("==============================================")
print("               YAW RATE")
print("==============================================")
print()

yaw = segment[yaw_col]

print("Minimum:", yaw.min(), "deg/s")
print("Maximum:", yaw.max(), "deg/s")
print("Average:", yaw.mean(), "deg/s")
print()

print("==============================================")
print("           LONGITUDINAL ACCEL")
print("==============================================")
print()

long_acc = segment[long_acc_col]

print("Minimum:", long_acc.min(), "g")
print("Maximum:", long_acc.max(), "g")
print("Average:", long_acc.mean(), "g")
print()

print("==============================================")
print("             LATERAL ACCEL")
print("==============================================")
print()

lat_acc = segment[lat_acc_col]

print("Minimum:", lat_acc.min(), "g")
print("Maximum:", lat_acc.max(), "g")
print("Average:", lat_acc.mean(), "g")
print()

print("==============================================")
print("            SAMPLE PERIOD")
print("==============================================")
print()

print("Minimum:", segment[sample_col].min())
print("Maximum:", segment[sample_col].max())
print("Average:", segment[sample_col].mean())
print()

print("==============================================")
print("              VALIDATION")
print("==============================================")

if speed_distance > 0:

    error = (
        abs(gps_distance - speed_distance)
        / speed_distance
        * 100
    )

    print()
    print("GPS distance:",
          round(gps_distance, 2), "m")

    print("Speed distance:",
          round(speed_distance, 2), "m")

    print("Difference:",
          round(error, 2), "%")

    print()

    if error < 20:
        print("STATUS: VALID SEGMENT")
    else:
        print("STATUS: SUSPICIOUS SEGMENT")

print()
print("==============================================")
print("       VEHICLE VALIDATION COMPLETE")
print("==============================================")