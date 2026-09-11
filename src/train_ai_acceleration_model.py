from pathlib import Path
import pandas as pd
import numpy as np

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

print(f"Smartphone dataset : {smart.shape}")
print(f"Vehicle dataset    : {vehicle.shape}")


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


# Smartphone columns

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


# Vehicle target

vehicle_acc_col = find_column(
    vehicle,
    "LONGITUDINAL ACCELERATION"
)


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
# NUMERIC ARRAYS
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


# Vehicle acceleration

vehicle_acc = pd.to_numeric(
    vehicle[vehicle_acc_col],
    errors="coerce"
).to_numpy()


# ============================================================
# GRAVITY-COMPENSATED ACCELERATION
# ============================================================

linear_acc = acc - gravity


# ============================================================
# MAGNITUDES
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
print("Creating acceleration features...")


for i in range(WINDOW - 1, len(smart)):

    # --------------------------------------------------------
    # Window
    # --------------------------------------------------------

    start = i - WINDOW + 1
    end = i + 1

    window_acc = acc[start:end]
    window_linear = linear_acc[start:end]
    window_gyro = gyro[start:end]
    window_mag = mag[start:end]

    # --------------------------------------------------------
    # Feature vector
    # --------------------------------------------------------

    f = []

    # Raw accelerometer XYZ
    for axis in range(3):

        f.extend(
            statistics(
                window_acc[:, axis]
            )
        )

    # Linear acceleration XYZ
    for axis in range(3):

        f.extend(
            statistics(
                window_linear[:, axis]
            )
        )

    # Gyroscope XYZ
    for axis in range(3):

        f.extend(
            statistics(
                window_gyro[:, axis]
            )
        )

    # Accelerometer magnitude
    f.extend(
        statistics(
            acc_mag[start:end]
        )
    )

    # Linear acceleration magnitude
    f.extend(
        statistics(
            linear_acc_mag[start:end]
        )
    )

    # Gyroscope magnitude
    f.extend(
        statistics(
            gyro_mag[start:end]
        )
    )

    # Magnetometer XYZ
    for axis in range(3):

        f.extend(
            statistics(
                window_mag[:, axis]
            )
        )

    # Magnetometer magnitude
    f.extend(
        statistics(
            mag_mag[start:end]
        )
    )

    # --------------------------------------------------------
    # Recent motion features
    # --------------------------------------------------------

    recent_5 = linear_acc_mag[
        max(0, i - 4):i + 1
    ]

    recent_10 = linear_acc_mag[
        max(0, i - 9):i + 1
    ]

    recent_gyro = gyro_mag[
        max(0, i - 4):i + 1
    ]

    f.extend([
        np.mean(recent_5),
        np.std(recent_5),

        np.mean(recent_10),
        np.std(recent_10),

        np.mean(recent_gyro),
        np.std(recent_gyro)
    ])

    features.append(f)

    targets.append(
        vehicle_acc[i]
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


# ============================================================
# REMOVE INVALID DATA
# ============================================================

valid = (
    np.isfinite(X).all(axis=1)
    &
    np.isfinite(y)
)

X = X[valid]
y = y[valid]


print()
print("=" * 55)
print("FEATURE DATASET")
print("=" * 55)

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
# TIME-BASED TRAIN / TEST SPLIT
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
# SCALE
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# MODEL
# ============================================================

print()
print("Training AI acceleration model...")

model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
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
print("=" * 55)
print("AI ACCELERATION MODEL RESULTS")
print("=" * 55)

print(
    f"MAE  : {mae:.4f} m/s²"
)

print(
    f"RMSE : {rmse:.4f} m/s²"
)

print(
    f"Actual test mean acceleration : "
    f"{np.mean(y_test):.4f} m/s²"
)

print(
    f"Predicted test mean acceleration : "
    f"{np.mean(prediction):.4f} m/s²"
)


# ============================================================
# SAVE
# ============================================================

model_path = (
    MODEL_DIR /
    "ai_acceleration_model.joblib"
)

scaler_path = (
    MODEL_DIR /
    "ai_acceleration_scaler.joblib"
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
print("=" * 55)
print("MODEL SAVED")
print("=" * 55)

print(model_path)
print(scaler_path)

print()
print("Training complete.")