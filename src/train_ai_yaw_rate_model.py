from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


def find_file(name):

    matches = list(DATA_DIR.rglob(name))

    if not matches:
        raise FileNotFoundError(
            f"{name} was not found inside {DATA_DIR}"
        )

    return matches[0]


SMARTPHONE_FILE = find_file("S-S1.csv")
VEHICLE_FILE = find_file("V-S1.csv")


# ============================================================
# LOAD DATA
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

print(
    f"Smartphone dataset : {smart.shape}"
)

print(
    f"Vehicle dataset    : {vehicle.shape}"
)


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(df, text):

    text = text.upper()

    for col in df.columns:

        if text in str(col).upper():
            return col

    raise KeyError(
        f"Column containing '{text}' not found"
    )


# ============================================================
# SMARTPHONE COLUMNS
# ============================================================

time_col = find_column(
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

gyro_yaw_col = find_column(
    smart,
    "GYROSCOPE Yaw"
)

gyro_pitch_col = find_column(
    smart,
    "GYROSCOPE Pitch"
)

gyro_roll_col = find_column(
    smart,
    "GYROSCOPE Roll"
)

mag_x_col = find_column(
    smart,
    "MAGNETIC FIELD X"
)

mag_y_col = find_column(
    smart,
    "MAGNETIC FIELD Y"
)

mag_z_col = find_column(
    smart,
    "MAGNETIC FIELD Z"
)


# ============================================================
# VEHICLE HEADING
# ============================================================

vehicle_heading_col = find_column(
    vehicle,
    "Heading"
)


# ============================================================
# TIME
# ============================================================

time = (
    pd.to_numeric(
        smart[time_col],
        errors="coerce"
    ).to_numpy()
    / 1000.0
)


# ============================================================
# SENSOR DATA
# ============================================================

acc = smart[
    [
        acc_x_col,
        acc_y_col,
        acc_z_col
    ]
].apply(
    pd.to_numeric,
    errors="coerce"
).to_numpy()


gravity = smart[
    [
        gravity_x_col,
        gravity_y_col,
        gravity_z_col
    ]
].apply(
    pd.to_numeric,
    errors="coerce"
).to_numpy()


gyro = smart[
    [
        gyro_yaw_col,
        gyro_pitch_col,
        gyro_roll_col
    ]
].apply(
    pd.to_numeric,
    errors="coerce"
).to_numpy()


mag = smart[
    [
        mag_x_col,
        mag_y_col,
        mag_z_col
    ]
].apply(
    pd.to_numeric,
    errors="coerce"
).to_numpy()


vehicle_heading = pd.to_numeric(
    vehicle[vehicle_heading_col],
    errors="coerce"
).to_numpy()


# ============================================================
# GRAVITY COMPENSATION
# ============================================================

linear_acc = acc - gravity


# ============================================================
# SENSOR MAGNITUDES
# ============================================================

acc_mag = np.linalg.norm(
    acc,
    axis=1
)

linear_acc_mag = np.linalg.norm(
    linear_acc,
    axis=1
)

gyro_mag = np.linalg.norm(
    gyro,
    axis=1
)

mag_mag = np.linalg.norm(
    mag,
    axis=1
)


# ============================================================
# CLEAN VEHICLE HEADING
# ============================================================

print()
print("Cleaning vehicle heading...")


# Convert heading to radians and unwrap it

heading_rad = np.deg2rad(
    vehicle_heading
)

heading_unwrapped = np.unwrap(
    heading_rad
)

heading_deg = np.rad2deg(
    heading_unwrapped
)


# ============================================================
# INITIAL YAW RATE
# ============================================================

raw_dt = np.diff(time)

raw_heading_change = np.diff(
    heading_deg
)

raw_yaw_rate = (
    raw_heading_change /
    raw_dt
)


# ============================================================
# DETECT BAD HEADING JUMPS
# ============================================================

MAX_REASONABLE_RATE = 20.0


bad = (
    ~np.isfinite(raw_yaw_rate)
    |
    (raw_dt <= 0)
    |
    (np.abs(raw_yaw_rate) >
     MAX_REASONABLE_RATE)
)


bad_sample_indices = np.where(
    bad
)[0] + 1


print(
    f"Detected suspicious heading samples : "
    f"{len(bad_sample_indices)}"
)

print(
    f"Percentage of samples              : "
    f"{100 * len(bad_sample_indices) / len(vehicle):.2f}%"
)


# ============================================================
# REMOVE BAD HEADING SAMPLES
# ============================================================

clean_heading = heading_deg.copy()

clean_heading[
    bad_sample_indices
] = np.nan


# ============================================================
# INTERPOLATE BAD SAMPLES
# ============================================================

clean_heading_series = pd.Series(
    clean_heading
)

clean_heading_series = (
    clean_heading_series
    .interpolate(
        method="linear"
    )
    .ffill()
    .bfill()
)

clean_heading = (
    clean_heading_series
    .to_numpy()
)


# ============================================================
# SMOOTH HEADING
# ============================================================

clean_heading_series = pd.Series(
    clean_heading
)

clean_heading_smooth = (
    clean_heading_series
    .rolling(
        window=5,
        center=True,
        min_periods=1
    )
    .mean()
    .to_numpy()
)


# ============================================================
# CALCULATE CLEAN YAW RATE
# ============================================================

clean_yaw_rate = np.full(
    len(vehicle),
    np.nan
)


for i in range(
    1,
    len(vehicle) - 1
):

    local_dt = (
        time[i + 1]
        -
        time[i - 1]
    )

    if (
        local_dt > 0.1
        and
        local_dt < 0.3
    ):

        clean_yaw_rate[i] = (
            clean_heading_smooth[i + 1]
            -
            clean_heading_smooth[i - 1]
        ) / local_dt


# ============================================================
# FINAL TARGET CLEANING
# ============================================================

valid_target = (
    np.isfinite(
        clean_yaw_rate
    )
    &
    (np.abs(clean_yaw_rate) <= 20)
)


print()
print("=" * 60)
print("CLEAN YAW-RATE TARGET")
print("=" * 60)

print(
    f"Minimum : "
    f"{np.nanmin(clean_yaw_rate):.4f} deg/s"
)

print(
    f"Maximum : "
    f"{np.nanmax(clean_yaw_rate):.4f} deg/s"
)

print(
    f"Mean    : "
    f"{np.nanmean(clean_yaw_rate):.4f} deg/s"
)

print(
    f"Median  : "
    f"{np.nanmedian(clean_yaw_rate):.4f} deg/s"
)

print(
    f"Valid targets : "
    f"{np.sum(valid_target)}"
)


# ============================================================
# FEATURE CREATION
# ============================================================

WINDOW = 20


def statistics(values):

    return [
        np.mean(values),
        np.std(values),
        np.min(values),
        np.max(values),
        np.median(values),
        np.ptp(values)
    ]


features = []
targets = []


print()
print("Creating yaw-rate features...")


for i in range(
    WINDOW - 1,
    len(smart)
):

    if not valid_target[i]:
        continue


    start = i - WINDOW + 1
    end = i + 1


    window_acc = (
        acc[start:end]
    )

    window_linear = (
        linear_acc[start:end]
    )

    window_gyro = (
        gyro[start:end]
    )

    window_mag = (
        mag[start:end]
    )


    f = []


    # --------------------------------------------------------
    # RAW ACCELEROMETER
    # --------------------------------------------------------

    for axis in range(3):

        f.extend(
            statistics(
                window_acc[:, axis]
            )
        )


    # --------------------------------------------------------
    # LINEAR ACCELERATION
    # --------------------------------------------------------

    for axis in range(3):

        f.extend(
            statistics(
                window_linear[:, axis]
            )
        )


    # --------------------------------------------------------
    # GYROSCOPE
    # --------------------------------------------------------

    for axis in range(3):

        f.extend(
            statistics(
                window_gyro[:, axis]
            )
        )


    # --------------------------------------------------------
    # MAGNETOMETER
    # --------------------------------------------------------

    for axis in range(3):

        f.extend(
            statistics(
                window_mag[:, axis]
            )
        )


    # --------------------------------------------------------
    # MAGNITUDES
    # --------------------------------------------------------

    f.extend(
        statistics(
            acc_mag[start:end]
        )
    )

    f.extend(
        statistics(
            linear_acc_mag[start:end]
        )
    )

    f.extend(
        statistics(
            gyro_mag[start:end]
        )
    )

    f.extend(
        statistics(
            mag_mag[start:end]
        )
    )


    # --------------------------------------------------------
    # RECENT MOTION
    # --------------------------------------------------------

    recent_5_linear = (
        linear_acc_mag[
            max(0, i - 4):i + 1
        ]
    )

    recent_10_linear = (
        linear_acc_mag[
            max(0, i - 9):i + 1
        ]
    )

    recent_5_gyro = (
        gyro_mag[
            max(0, i - 4):i + 1
        ]
    )


    f.extend([
        np.mean(recent_5_linear),
        np.std(recent_5_linear),

        np.mean(recent_10_linear),
        np.std(recent_10_linear),

        np.mean(recent_5_gyro),
        np.std(recent_5_gyro)
    ])


    if np.isfinite(f).all():

        features.append(f)

        targets.append(
            clean_yaw_rate[i]
        )


# ============================================================
# ARRAYS
# ============================================================

X = np.asarray(
    features,
    dtype=float
)

y = np.asarray(
    targets,
    dtype=float
)


print()
print("=" * 60)
print("FEATURE DATASET")
print("=" * 60)

print(
    f"Feature matrix : {X.shape}"
)

print(
    f"Target shape   : {y.shape}"
)

print(
    f"Number features: {X.shape[1]}"
)


# ============================================================
# TIME-BASED SPLIT
# ============================================================

split = int(
    len(X) * 0.80
)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]


print()
print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = (
    scaler.fit_transform(
        X_train
    )
)

X_test_scaled = (
    scaler.transform(
        X_test
    )
)


# ============================================================
# MODEL
# ============================================================

print()
print("Training AI yaw-rate model...")


model = MLPRegressor(
    hidden_layer_sizes=(
        64,
        32,
        16
    ),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=20,
    batch_size=128,
    random_state=42
)


model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# TEST
# ============================================================

prediction = model.predict(
    X_test_scaled
)


mae = mean_absolute_error(
    y_test,
    prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction
    )
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("CLEAN AI YAW-RATE MODEL RESULTS")
print("=" * 60)

print(
    f"MAE  : {mae:.4f} deg/s"
)

print(
    f"RMSE : {rmse:.4f} deg/s"
)

print(
    f"Actual test mean yaw-rate : "
    f"{np.mean(y_test):.4f} deg/s"
)

print(
    f"Predicted test mean yaw-rate : "
    f"{np.mean(prediction):.4f} deg/s"
)


# ============================================================
# SAVE
# ============================================================

model_path = (
    MODEL_DIR /
    "ai_yaw_rate_model.joblib"
)

scaler_path = (
    MODEL_DIR /
    "ai_yaw_rate_scaler.joblib"
)


dump(
    model,
    model_path
)

dump(
    scaler,
    scaler_path
)


print()
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(model_path)
print(scaler_path)

print()
print("Training complete.")