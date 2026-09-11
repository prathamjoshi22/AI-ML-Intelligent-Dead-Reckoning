import pandas as pd
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
speed_column = find_column("GPS SPEED")

df["time_seconds"] = pd.to_numeric(
    df[time_column], errors="coerce"
) / 1000

df["speed_kmh"] = pd.to_numeric(
    df[speed_column], errors="coerce"
)

window = 60

results = []

# Check every 10 seconds instead of every 0.1 seconds
for start in range(
    int(df["time_seconds"].min()),
    int(df["time_seconds"].max() - window),
    10
):
    end = start + window

    section = df[
        (df["time_seconds"] >= start) &
        (df["time_seconds"] <= end)
    ]

    if len(section) < 100:
        continue

    average_speed = section["speed_kmh"].mean()
    maximum_speed = section["speed_kmh"].max()

    results.append(
        {
            "start": start,
            "end": end,
            "average_speed": average_speed,
            "maximum_speed": maximum_speed
        }
    )

results = sorted(
    results,
    key=lambda x: x["average_speed"],
    reverse=True
)

print("======================================")
print("       BEST GNSS BLACKOUT SECTIONS")
print("======================================")
print()

print("Top 10 candidate 60-second intervals:")
print()

for i, result in enumerate(results[:10], 1):

    print(
        f"{i}. Start: {result['start']}s | "
        f"End: {result['end']}s | "
        f"Average speed: {result['average_speed']:.2f} km/h | "
        f"Maximum speed: {result['maximum_speed']:.2f} km/h"
    )

print()
print("======================================")
print("Candidate search completed")
print("======================================")