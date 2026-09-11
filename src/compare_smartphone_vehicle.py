import pandas as pd
import numpy as np
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "Data"

smartphone_files = list(data_folder.rglob("S-S1.csv"))
vehicle_files = list(data_folder.rglob("V-S1.csv"))

if not smartphone_files:
    raise FileNotFoundError("S-S1.csv not found.")

if not vehicle_files:
    raise FileNotFoundError("V-S1.csv not found.")

smartphone = pd.read_csv(
    smartphone_files[0],
    encoding="cp1252"
)

vehicle = pd.read_csv(
    vehicle_files[0],
    encoding="cp1252"
)

smartphone.columns = smartphone.columns.str.strip()
vehicle.columns = vehicle.columns.str.strip()

smart_lat = smartphone["GPS LATITUDE (degrees)"]
smart_lon = smartphone["GPS LONGITUDE (degrees)"]

vehicle_lat = vehicle["Latitude (degrees)"]
vehicle_lon = vehicle["Longitude (degrees)"]

print("==============================================")
print("      SMARTPHONE / VEHICLE GPS COMPARISON")
print("==============================================")
print()

print("Rows:")
print("Smartphone:", len(smartphone))
print("Vehicle   :", len(vehicle))
print()

# Compare GPS positions at several row locations

indices = [
    0,
    10000,
    20000,
    30000,
    40000,
    50000
]

print("==============================================")
print("             POSITION COMPARISON")
print("==============================================")
print()

for index in indices:

    if index >= len(smartphone):
        continue

    slat = smart_lat.iloc[index]
    slon = smart_lon.iloc[index]

    vlat = vehicle_lat.iloc[index]
    vlon = vehicle_lon.iloc[index]

    lat_difference = abs(slat - vlat)
    lon_difference = abs(slon - vlon)

    lat_meters = lat_difference * 111320

    lon_meters = (
        lon_difference *
        111320 *
        np.cos(np.radians(vlat))
    )

    distance = np.sqrt(
        lat_meters**2 +
        lon_meters**2
    )

    print("Row:", index)

    print(
        "Smartphone:",
        round(slat, 6),
        round(slon, 6)
    )

    print(
        "Vehicle   :",
        round(vlat, 6),
        round(vlon, 6)
    )

    print(
        "Difference:",
        round(distance, 2),
        "meters"
    )

    print()

print("==============================================")
print("          SELECTED OUTAGE COMPARISON")
print("==============================================")
print()

start = 3562
end = 3622

smart_time = smartphone["TIME SINCE START (ms)"] / 1000

start_index = (
    (smart_time - start).abs()
).idxmin()

end_index = (
    (smart_time - end).abs()
).idxmin()

print("Smartphone start index:", start_index)
print("Smartphone end index  :", end_index)
print()

print("Corresponding vehicle rows:")
print(start_index, "→", end_index)
print()

print("==============================================")
print("                COMPLETE")
print("==============================================")