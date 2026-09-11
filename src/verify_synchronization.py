import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "Data"

smartphone_files = list(data_folder.rglob("S-S1.csv"))
vehicle_files = list(data_folder.rglob("V-S1.csv"))

if not smartphone_files:
    raise FileNotFoundError("S-S1.csv not found.")

if not vehicle_files:
    raise FileNotFoundError("V-S1.csv not found.")

smartphone = pd.read_csv(smartphone_files[0], encoding="cp1252")
vehicle = pd.read_csv(vehicle_files[0], encoding="cp1252")

smartphone.columns = smartphone.columns.str.strip()
vehicle.columns = vehicle.columns.str.strip()

smart_time_col = "TIME SINCE START (ms)"
vehicle_time_col = "Time Since Start of Day (seconds)"

smartphone_time = smartphone[smart_time_col] / 1000
vehicle_time = vehicle[vehicle_time_col]

print("==============================================")
print("       SMARTPHONE / VEHICLE SYNCHRONIZATION")
print("==============================================")
print()

print("Smartphone rows:", len(smartphone))
print("Vehicle rows   :", len(vehicle))
print()

print("Smartphone time:")
print("Start:", smartphone_time.iloc[0])
print("End  :", smartphone_time.iloc[-1])
print()

print("Vehicle time:")
print("Start:", vehicle_time.iloc[0])
print("End  :", vehicle_time.iloc[-1])
print()

offset = vehicle_time.iloc[0] - smartphone_time.iloc[0]

print("==============================================")
print("                 TIME OFFSET")
print("==============================================")
print()

print("Vehicle - Smartphone offset:")
print(offset, "seconds")
print()

print("Expected offset approximately:")
print(vehicle_time.iloc[0], "seconds")
print()

print("==============================================")
print("           SAMPLE INTERVAL CHECK")
print("==============================================")
print()

smart_dt = smartphone_time.diff().dropna()
vehicle_dt = vehicle_time.diff().dropna()

print("Smartphone average dt:",
      smart_dt.mean(), "seconds")

print("Vehicle average dt:",
      vehicle_dt.mean(), "seconds")

print()

print("Smartphone minimum dt:",
      smart_dt.min())

print("Smartphone maximum dt:",
      smart_dt.max())

print()

print("Vehicle minimum dt:",
      vehicle_dt.min())

print("Vehicle maximum dt:",
      vehicle_dt.max())

print()

print("==============================================")
print("              SYNCHRONIZATION")
print("==============================================")

if abs(offset - 32869) < 1:
    print()
    print("STATUS: SYNCHRONIZATION LOOKS VALID")
else:
    print()
    print("STATUS: OFFSET NEEDS FURTHER CHECKING")

print()
print("==============================================")
print("       SYNCHRONIZATION CHECK COMPLETE")
print("==============================================")