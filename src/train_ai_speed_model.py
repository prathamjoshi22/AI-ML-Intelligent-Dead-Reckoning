# ============================================================
# IMPROVED AI SPEED MODEL
# SIH 2026 - GNSS DENIED POSITIONING
# ============================================================

import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# 2. FIND DATASETS
# ============================================================

smartphone_file = next(
    DATA_DIR.rglob("S-S1.csv")
)

vehicle_file = next(
    DATA_DIR.rglob("V-S1.csv")
)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("Loading datasets...")

smart = pd.read_csv(
    smartphone_file,
    encoding="cp1252"
)

vehicle = pd.read_csv(
    vehicle_file,
    encoding="cp1252"
)

print(
    f"Smartphone dataset: {smart.shape}"
)

print(
    f"Vehicle dataset:    {vehicle.shape}"
)


# ============================================================
# 4. COLUMN FINDER
# ============================================================

def find_column(df, text):

    matches = [
        col
        for col in df.columns
        if text.lower() in col.lower()
    ]

    if not matches:

        raise ValueError(
            f"Column containing '{text}' not found."
        )

    return matches[0]


# ============================================================
# 5. SMARTPHONE COLUMNS
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


# ============================================================
# 6. VEHICLE SPEED COLUMN
# ============================================================

vehicle_speed_col = find_column(
    vehicle,
    "Velocity"
)


# ============================================================
# 7. EXTRACT SENSOR DATA
# ============================================================

time = (
    smart[time_col]
    .astype(float)
    .values
    / 1000.0
)


acc_x = (
    smart[acc_x_col]
    .astype(float)
    .values
)

acc_y = (
    smart[acc_y_col]
    .astype(float)
    .values
)

acc_z = (
    smart[acc_z_col]
    .astype(float)
    .values
)


gravity_x = (
    smart[gravity_x_col]
    .astype(float)
    .values
)

gravity_y = (
    smart[gravity_y_col]
    .astype(float)
    .values
)

gravity_z = (
    smart[gravity_z_col]
    .astype(float)
    .values
)


gyro_yaw = (
    smart[gyro_yaw_col]
    .astype(float)
    .values
)

gyro_pitch = (
    smart[gyro_pitch_col]
    .astype(float)
    .values
)

gyro_roll = (
    smart[gyro_roll_col]
    .astype(float)
    .values
)


# ============================================================
# 8. GRAVITY-COMPENSATED ACCELERATION
# ============================================================

linear_x = (
    acc_x - gravity_x
)

linear_y = (
    acc_y - gravity_y
)

linear_z = (
    acc_z - gravity_z
)


# ============================================================
# 9. MAGNITUDES
# ============================================================

acc_magnitude = np.sqrt(
    acc_x ** 2
    + acc_y ** 2
    + acc_z ** 2
)

linear_acc_magnitude = np.sqrt(
    linear_x ** 2
    + linear_y ** 2
    + linear_z ** 2
)

gyro_magnitude = np.sqrt(
    gyro_yaw ** 2
    + gyro_pitch ** 2
    + gyro_roll ** 2
)


# ============================================================
# 10. FEATURE FUNCTION
# ============================================================

def statistics(arr):

    return [
        np.mean(arr),
        np.std(arr),
        np.min(arr),
        np.max(arr),
        np.median(arr),
        np.ptp(arr)
    ]


def create_features(i):

    start = i - 19

    window_slice = slice(
        start,
        i + 1
    )

    features = []


    # --------------------------------------------------------
    # Raw accelerometer
    # --------------------------------------------------------

    for arr in [
        acc_x,
        acc_y,
        acc_z
    ]:

        features.extend(
            statistics(
                arr[window_slice]
            )
        )


    # --------------------------------------------------------
    # Gravity compensated acceleration
    # --------------------------------------------------------

    for arr in [
        linear_x,
        linear_y,
        linear_z
    ]:

        features.extend(
            statistics(
                arr[window_slice]
            )
        )


    # --------------------------------------------------------
    # Gyroscope
    # --------------------------------------------------------

    for arr in [
        gyro_yaw,
        gyro_pitch,
        gyro_roll
    ]:

        features.extend(
            statistics(
                arr[window_slice]
            )
        )


    # --------------------------------------------------------
    # Magnitudes
    # --------------------------------------------------------

    for arr in [
        acc_magnitude,
        linear_acc_magnitude,
        gyro_magnitude
    ]:

        features.extend(
            statistics(
                arr[window_slice]
            )
        )


    # --------------------------------------------------------
    # Recent motion features
    # --------------------------------------------------------

    last_5 = slice(
        i - 4,
        i + 1
    )

    last_10 = slice(
        i - 9,
        i + 1
    )


    features.extend([

        np.mean(
            linear_acc_magnitude[last_5]
        ),

        np.std(
            linear_acc_magnitude[last_5]
        ),

        np.mean(
            linear_acc_magnitude[last_10]
        ),

        np.std(
            linear_acc_magnitude[last_10]
        ),

        np.mean(
            gyro_magnitude[last_5]
        ),

        np.std(
            gyro_magnitude[last_5]
        )

    ])


    return np.asarray(
        features,
        dtype=np.float64
    )


# ============================================================
# 11. BUILD FEATURE MATRIX
# ============================================================

print()
print("Building AI features...")

X = []
y = []

window_size = 20


for i in range(
    window_size - 1,
    len(smart)
):

    try:

        features = create_features(i)

        target = float(
            vehicle[
                vehicle_speed_col
            ].iloc[i]
        )

        if np.isfinite(
            features
        ).all() and np.isfinite(target):

            X.append(
                features
            )

            y.append(
                target
            )

    except Exception:

        continue


X = np.asarray(
    X
)

y = np.asarray(
    y
)


print()
print("======================================")
print("FEATURE DATASET")
print("======================================")

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
# 12. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

# IMPORTANT:
# We do NOT randomly shuffle the driving data.
#
# Earlier random splitting could allow very similar
# neighbouring samples to appear in both train and test.
#
# Here:
#
# First 80% -> training
# Last 20%  -> testing


split_index = int(
    len(X) * 0.80
)


X_train = X[
    :split_index
]

X_test = X[
    split_index:
]

y_train = y[
    :split_index
]

y_test = y[
    split_index:
]


print()
print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# 13. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 14. TRAIN MLP
# ============================================================

print()
print("Training improved AI speed model...")

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

    random_state=42,

    early_stopping=True,

    validation_fraction=0.10,

    n_iter_no_change=20,

    batch_size=128

)


model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 15. PREDICTION
# ============================================================

predictions = model.predict(
    X_test_scaled
)


predictions = np.clip(
    predictions,
    0,
    None
)


# ============================================================
# 16. METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


# ============================================================
# 17. RESULTS
# ============================================================

print()
print("======================================")
print("IMPROVED AI SPEED MODEL RESULTS")
print("======================================")

print(
    f"MAE  : {mae:.2f} km/h"
)

print(
    f"RMSE : {rmse:.2f} km/h"
)

print(
    f"Actual test average speed : "
    f"{np.mean(y_test):.2f} km/h"
)

print(
    f"Predicted test average speed : "
    f"{np.mean(predictions):.2f} km/h"
)


# ============================================================
# 18. SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "ai_speed_model.joblib"
)

scaler_path = (
    MODEL_DIR
    / "ai_speed_scaler.joblib"
)


joblib.dump(
    model,
    model_path
)

joblib.dump(
    scaler,
    scaler_path
)


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print()
print("======================================")
print("MODEL SAVED")
print("======================================")

print(
    model_path
)

print(
    scaler_path
)

print()
print(
    "AI SPEED MODEL TRAINING COMPLETE."
)