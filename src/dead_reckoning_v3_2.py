import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.signal import butter, filtfilt


# ============================================================
# CONFIGURATION
# ============================================================

START_TIME = 3562.0
DURATION = 60.0

# Seconds before GNSS outage used for bias estimation
BIAS_WINDOW = 5.0

# Low-pass filter cutoff
FILTER_CUTOFF_HZ = 1.0

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_csv(filename):
    matches = list(DATA_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} inside {DATA_DIR}"
        )

    return matches[0]


def find_column(df, text):
    text = text.lower()

    for col in df.columns:
        if text in col.lower():
            return col

    raise KeyError(
        f"Could not find column containing: {text}"
    )


def latlon_to_local(lat, lon, lat0, lon0):
    R = 6371000.0

    north = (
        np.deg2rad(lat - lat0)
        * R
    )

    east = (
        np.deg2rad(lon - lon0)
        * R
        * np.cos(np.deg2rad(lat0))
    )

    return east, north


def low_pass_filter(data, cutoff, sample_rate):
    nyquist = 0.5 * sample_rate

    normalized_cutoff = cutoff / nyquist

    b, a = butter(
        2,
        normalized_cutoff,
        btype="low"
    )

    return filtfilt(
        b,
        a,
        data,
        axis=0
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("GNSS-DENIED DEAD RECKONING BASELINE V3.2")
print("BIAS CORRECTION + LOW-PASS FILTER TEST")
print("=" * 65)


# ============================================================
# LOAD DATA
# ============================================================

smart_path = find_csv("S-S1.csv")
vehicle_path = find_csv("V-S1.csv")

print("\nSmartphone file:")
print(smart_path)

print("\nVehicle file:")
print(vehicle_path)

smart = pd.read_csv(
    smart_path,
    encoding="cp1252"
)

vehicle = pd.read_csv(
    vehicle_path,
    encoding="cp1252"
)

print("\nSmartphone shape:", smart.shape)
print("Vehicle shape:", vehicle.shape)


# ============================================================
# FIND SMARTPHONE COLUMNS
# ============================================================

smart_time_col = find_column(
    smart,
    "TIME SINCE START"
)

acc_x_col = find_column(
    smart,
    "ACCELEROMETER X"
)

acc_y_col = find_column(
    smart,
    "ACCELEROMETER Y"
)

acc_z_col = find_column(
    smart,
    "ACCELEROMETER Z"
)

gravity_x_col = find_column(
    smart,
    "GRAVITY X"
)

gravity_y_col = find_column(
    smart,
    "GRAVITY Y"
)

gravity_z_col = find_column(
    smart,
    "GRAVITY Z"
)

yaw_col = find_column(
    smart,
    "ORIENTATION (Yaw)"
)

pitch_col = find_column(
    smart,
    "ORIENTATION (Pitch)"
)

roll_col = find_column(
    smart,
    "ORIENTATION (Roll"
)


# ============================================================
# FIND VEHICLE COLUMNS
# ============================================================

vehicle_time_col = find_column(
    vehicle,
    "Time"
)

vehicle_lat_col = find_column(
    vehicle,
    "Latitude"
)

vehicle_lon_col = find_column(
    vehicle,
    "Longitude"
)

vehicle_speed_col = find_column(
    vehicle,
    "Velocity"
)

vehicle_heading_col = find_column(
    vehicle,
    "Heading"
)


# ============================================================
# TIME
# ============================================================

smart_time = (
    pd.to_numeric(
        smart[smart_time_col],
        errors="coerce"
    ) / 1000.0
)

vehicle_time = pd.to_numeric(
    vehicle[vehicle_time_col],
    errors="coerce"
)


# ============================================================
# OUTAGE WINDOW
# ============================================================

end_time = START_TIME + DURATION

smart_mask = (
    (smart_time >= START_TIME)
    &
    (smart_time <= end_time)
)

smart_segment = smart.loc[
    smart_mask
].copy()

smart_segment["relative_time"] = (
    smart_time.loc[
        smart_mask
    ].values - START_TIME
)

print("\nSelected smartphone samples:")
print(len(smart_segment))

if len(smart_segment) < 10:
    raise RuntimeError(
        "Not enough smartphone samples."
    )


# ============================================================
# MATCH VEHICLE ROWS
# ============================================================

start_index = smart_segment.index[0]
end_index = smart_segment.index[-1]

vehicle_segment = vehicle.iloc[
    start_index:end_index + 1
].copy()

print("Vehicle samples:", len(vehicle_segment))

if len(vehicle_segment) != len(smart_segment):
    raise RuntimeError(
        "Smartphone and vehicle segment lengths do not match."
    )


# ============================================================
# TIME STEP
# ============================================================

t = smart_segment[
    "relative_time"
].to_numpy()

dt = np.diff(
    t,
    prepend=t[0]
)

dt[0] = np.median(
    np.diff(t)
)

sample_rate = 1.0 / np.median(dt)

print("\nAverage dt:", np.mean(dt))
print("Sample rate:", sample_rate, "Hz")


# ============================================================
# CONTROLLED INITIALIZATION
# ============================================================

initial_speed_kmh = float(
    pd.to_numeric(
        vehicle_segment[
            vehicle_speed_col
        ],
        errors="coerce"
    ).iloc[0]
)

initial_heading_deg = float(
    pd.to_numeric(
        vehicle_segment[
            vehicle_heading_col
        ],
        errors="coerce"
    ).iloc[0]
)

initial_lat = float(
    pd.to_numeric(
        vehicle_segment[
            vehicle_lat_col
        ],
        errors="coerce"
    ).iloc[0]
)

initial_lon = float(
    pd.to_numeric(
        vehicle_segment[
            vehicle_lon_col
        ],
        errors="coerce"
    ).iloc[0]
)


# ============================================================
# INITIAL VELOCITY VECTOR
# ============================================================

initial_speed = (
    initial_speed_kmh / 3.6
)

heading_rad = np.deg2rad(
    initial_heading_deg
)

initial_velocity_east = (
    initial_speed
    * np.sin(heading_rad)
)

initial_velocity_north = (
    initial_speed
    * np.cos(heading_rad)
)

initial_velocity = np.array([
    initial_velocity_east,
    initial_velocity_north
])

print("\n")
print("=" * 65)
print("CONTROLLED INITIALIZATION")
print("=" * 65)

print(
    f"Initial speed   : "
    f"{initial_speed_kmh:.3f} km/h"
)

print(
    f"Initial heading : "
    f"{initial_heading_deg:.3f} degrees"
)

print(
    f"Initial latitude: "
    f"{initial_lat:.8f}"
)

print(
    f"Initial longitude: "
    f"{initial_lon:.8f}"
)


# ============================================================
# LOAD SENSOR DATA
# ============================================================

acc = smart_segment[
    [
        acc_x_col,
        acc_y_col,
        acc_z_col
    ]
].to_numpy(
    dtype=float
)

gravity = smart_segment[
    [
        gravity_x_col,
        gravity_y_col,
        gravity_z_col
    ]
].to_numpy(
    dtype=float
)

yaw = pd.to_numeric(
    smart_segment[yaw_col],
    errors="coerce"
).to_numpy(
    dtype=float
)

pitch = pd.to_numeric(
    smart_segment[pitch_col],
    errors="coerce"
).to_numpy(
    dtype=float
)

roll = pd.to_numeric(
    smart_segment[roll_col],
    errors="coerce"
).to_numpy(
    dtype=float
)


# ============================================================
# GRAVITY COMPENSATION
# ============================================================

linear_acc = (
    acc - gravity
)


# ============================================================
# TRANSFORM DEVICE ACCELERATION
# INTO WORLD COORDINATES
# ============================================================

world_acc = np.zeros_like(
    linear_acc
)

for i in range(
    len(linear_acc)
):

    rotation = Rotation.from_euler(
        "ZYX",
        [
            yaw[i],
            pitch[i],
            roll[i]
        ],
        degrees=True
    )

    world_acc[i] = rotation.apply(
        linear_acc[i]
    )


# ============================================================
# HORIZONTAL ACCELERATION
# ============================================================

horizontal_acc = world_acc[
    :,
    [0, 1]
]


# ============================================================
# ESTIMATE PRE-OUTAGE BIAS
#
# NOTE:
# This is a diagnostic estimate.
# We assume the vehicle is approximately
# steady during the few seconds immediately
# before the selected outage.
# ============================================================

bias_start_time = (
    START_TIME - BIAS_WINDOW
)

bias_mask = (
    (smart_time >= bias_start_time)
    &
    (smart_time < START_TIME)
)

bias_indices = smart.index[
    bias_mask
]

print("\nBias estimation samples:")

if len(bias_indices) < 10:

    print(
        "Not enough pre-outage samples."
    )

    bias = np.array([
        0.0,
        0.0
    ])

else:

    pre_acc = smart.loc[
        bias_indices,
        [
            acc_x_col,
            acc_y_col,
            acc_z_col
        ]
    ].to_numpy(
        dtype=float
    )

    pre_gravity = smart.loc[
        bias_indices,
        [
            gravity_x_col,
            gravity_y_col,
            gravity_z_col
        ]
    ].to_numpy(
        dtype=float
    )

    pre_yaw = pd.to_numeric(
        smart.loc[
            bias_indices,
            yaw_col
        ],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    pre_pitch = pd.to_numeric(
        smart.loc[
            bias_indices,
            pitch_col
        ],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    pre_roll = pd.to_numeric(
        smart.loc[
            bias_indices,
            roll_col
        ],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    pre_linear_acc = (
        pre_acc - pre_gravity
    )

    pre_world_acc = np.zeros_like(
        pre_linear_acc
    )

    for i in range(
        len(pre_linear_acc)
    ):

        rotation = Rotation.from_euler(
            "ZYX",
            [
                pre_yaw[i],
                pre_pitch[i],
                pre_roll[i]
            ],
            degrees=True
        )

        pre_world_acc[i] = rotation.apply(
            pre_linear_acc[i]
        )

    pre_horizontal = pre_world_acc[
        :,
        [0, 1]
    ]

    # Median is more robust than mean
    # against sudden acceleration spikes.

    bias = np.median(
        pre_horizontal,
        axis=0
    )

    print(
        f"Bias samples used: "
        f"{len(pre_horizontal)}"
    )


print(
    f"Estimated East bias  : "
    f"{bias[0]:.6f} m/s²"
)

print(
    f"Estimated North bias : "
    f"{bias[1]:.6f} m/s²"
)


# ============================================================
# APPLY BIAS CORRECTION
# ============================================================

corrected_acc = (
    horizontal_acc - bias
)


# ============================================================
# LOW-PASS FILTER
# ============================================================

filtered_acc = low_pass_filter(
    corrected_acc,
    FILTER_CUTOFF_HZ,
    sample_rate
)


# ============================================================
# DEAD RECKONING
# ============================================================

velocity = initial_velocity.copy()

position = np.array([
    0.0,
    0.0
])

positions = [
    position.copy()
]

for i in range(
    1,
    len(filtered_acc)
):

    acceleration = filtered_acc[i]

    # Acceleration -> velocity

    velocity += (
        acceleration
        * dt[i]
    )

    # Velocity -> position

    position += (
        velocity
        * dt[i]
    )

    positions.append(
        position.copy()
    )

positions = np.array(
    positions
)


# ============================================================
# VEHICLE GROUND TRUTH
# ============================================================

vehicle_lat = pd.to_numeric(
    vehicle_segment[
        vehicle_lat_col
    ],
    errors="coerce"
).to_numpy()

vehicle_lon = pd.to_numeric(
    vehicle_segment[
        vehicle_lon_col
    ],
    errors="coerce"
).to_numpy()

truth_east, truth_north = latlon_to_local(
    vehicle_lat,
    vehicle_lon,
    vehicle_lat[0],
    vehicle_lon[0]
)


# ============================================================
# GROUND TRUTH DISTANCE
# ============================================================

truth_endpoint_distance = np.sqrt(
    truth_east[-1] ** 2
    +
    truth_north[-1] ** 2
)

truth_travel_distance = np.sum(
    np.sqrt(
        np.diff(truth_east) ** 2
        +
        np.diff(truth_north) ** 2
    )
)


# ============================================================
# DR METRICS
# ============================================================

dr_final_east = positions[-1, 0]

dr_final_north = positions[-1, 1]

dr_displacement = np.sqrt(
    dr_final_east ** 2
    +
    dr_final_north ** 2
)

final_error = np.sqrt(
    (
        dr_final_east
        -
        truth_east[-1]
    ) ** 2
    +
    (
        dr_final_north
        -
        truth_north[-1]
    ) ** 2
)

drift_percentage = (
    final_error
    /
    truth_travel_distance
) * 100


# ============================================================
# FINAL SPEED
# ============================================================

final_speed = np.sqrt(
    velocity[0] ** 2
    +
    velocity[1] ** 2
)

final_speed_kmh = (
    final_speed * 3.6
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("V3.2 RESULTS")
print("=" * 65)

print(
    f"Ground truth endpoint displacement : "
    f"{truth_endpoint_distance:.2f} m"
)

print(
    f"Ground truth traveled distance     : "
    f"{truth_travel_distance:.2f} m"
)

print(
    f"Dead reckoning displacement        : "
    f"{dr_displacement:.2f} m"
)

print(
    f"Final position error               : "
    f"{final_error:.2f} m"
)

print(
    f"Drift relative to traveled path   : "
    f"{drift_percentage:.2f}%"
)

print(
    f"Final DR speed                     : "
    f"{final_speed_kmh:.2f} km/h"
)


# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame({

    "time": t,

    "truth_east": truth_east,

    "truth_north": truth_north,

    "dr_east": positions[:, 0],

    "dr_north": positions[:, 1]

})

result_path = (
    BASE_DIR
    /
    "v3_2_results.csv"
)

result_df.to_csv(
    result_path,
    index=False
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(12, 8)
)

plt.plot(
    truth_east,
    truth_north,
    label="Vehicle Ground Truth",
    linewidth=2
)

plt.plot(
    positions[:, 0],
    positions[:, 1],
    label="IMU Dead Reckoning V3.2",
    linewidth=2
)

plt.scatter(
    0,
    0,
    s=100,
    label="Start"
)

plt.xlabel(
    "East (meters)"
)

plt.ylabel(
    "North (meters)"
)

plt.title(
    "GNSS-Denied Dead Reckoning Baseline V3.2\n"
    "Bias Correction + Low-Pass Filtering"
)

plt.grid(True)

plt.axis(
    "equal"
)

plt.legend()

plt.tight_layout()


# ============================================================
# SAVE GRAPH
# ============================================================

plot_path = (
    BASE_DIR
    /
    "v3_2_bias_filtered.png"
)

plt.savefig(
    plot_path,
    dpi=200
)

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 65)
print("FILES SAVED")
print("=" * 65)

print(
    f"Results CSV : {result_path}"
)

print(
    f"Graph       : {plot_path}"
)

print("\n")
print("V3.2 TEST COMPLETE.")