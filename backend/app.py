from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import math
import traceback


# ============================================================
# NAV-X BACKEND
# AI-ML Based Intelligent Dead Reckoning System
# SIH 2026 - SIH26168
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

MODEL_DIR = PROJECT_DIR / "models"
SRC_DIR = PROJECT_DIR / "src"

FUSION_FILE = SRC_DIR / "ai_fusion_results.csv"


# ============================================================
# GLOBAL DATA
# ============================================================

speed_model = None
speed_scaler = None

acceleration_model = None
acceleration_scaler = None

yaw_model = None
yaw_scaler = None

fusion_data = None


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:
        return default


def find_column(df, keywords):

    if isinstance(keywords, str):
        keywords = [keywords]

    for column in df.columns:

        name = str(column).lower()

        if all(
            keyword.lower() in name
            for keyword in keywords
        ):
            return column

    return None


def find_any_column(df, names):

    for name in names:

        column = find_column(df, name)

        if column is not None:
            return column

    return None


# ============================================================
# LOAD MODEL SAFELY
# ============================================================

def safe_load_model(path, label):

    if not path.exists():

        print(f"⚠ {label} not found")
        print(path)

        return None

    try:

        model = joblib.load(path)

        print(f"✓ {label} loaded")

        return model

    except Exception as error:

        print(f"⚠ Could not load {label}")

        print(error)

        print(
            "Dashboard will continue without this model."
        )

        return None


def load_models():

    global speed_model
    global speed_scaler

    global acceleration_model
    global acceleration_scaler

    global yaw_model
    global yaw_scaler

    print()
    print("=" * 65)
    print("LOADING NAV-X AI MODELS")
    print("=" * 65)

    speed_model = safe_load_model(
        MODEL_DIR / "ai_speed_model.joblib",
        "AI speed model"
    )

    speed_scaler = safe_load_model(
        MODEL_DIR / "ai_speed_scaler.joblib",
        "AI speed scaler"
    )

    acceleration_model = safe_load_model(
        MODEL_DIR / "ai_acceleration_model.joblib",
        "AI acceleration model"
    )

    acceleration_scaler = safe_load_model(
        MODEL_DIR / "ai_acceleration_scaler.joblib",
        "AI acceleration scaler"
    )

    yaw_model = safe_load_model(
        MODEL_DIR / "ai_yaw_rate_model.joblib",
        "AI yaw-rate model"
    )

    yaw_scaler = safe_load_model(
        MODEL_DIR / "ai_yaw_rate_scaler.joblib",
        "AI yaw-rate scaler"
    )

    print("=" * 65)


# ============================================================
# LOAD FUSION RESULTS
# ============================================================

def load_fusion_results():

    global fusion_data

    print()
    print("Loading fusion results...")

    if not FUSION_FILE.exists():

        print("⚠ ai_fusion_results.csv not found")

        return

    try:

        fusion_data = pd.read_csv(
            FUSION_FILE
        )

        print(
            f"✓ Fusion results loaded: "
            f"{len(fusion_data)} rows"
        )

        print(
            "Available columns:"
        )

        print(
            list(fusion_data.columns)
        )

    except Exception as error:

        print(
            "⚠ Could not load fusion results"
        )

        print(error)

        fusion_data = None


# ============================================================
# INITIALIZE
# ============================================================

load_models()
load_fusion_results()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({

        "name": "NAV-X",

        "description":
            "AI-ML Based Intelligent Dead Reckoning System",

        "problem_statement":
            "SIH26168",

        "organization":
            "ISRO",

        "theme":
            "Smart Vehicles",

        "status":
            "online"

    })


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({

        "status": "online",

        "backend": "Flask",

        "system": "NAV-X",

        "problem_statement": "SIH26168",

        "fusion_data_loaded":
            fusion_data is not None,

        "fusion_rows":
            0 if fusion_data is None
            else len(fusion_data)

    })


# ============================================================
# SYSTEM STATUS
# ============================================================

@app.route("/api/status")
def status():

    return jsonify({

        "system": "NAV-X",

        "gnss": "AVAILABLE",

        "navigation_mode":
            "GNSS + INS",

        "ai_status":
            "READY",

        "dead_reckoning":
            "READY",

        "map_matching":
            "READY",

        "nhc":
            "READY",

        "backend":
            "ONLINE"

    })


# ============================================================
# MODEL STATUS
# ============================================================

@app.route("/api/models")
def models():

    return jsonify({

        "speed_model":
            speed_model is not None,

        "acceleration_model":
            acceleration_model is not None,

        "yaw_rate_model":
            yaw_model is not None

    })


# ============================================================
# PERFORMANCE METRICS
# ============================================================

@app.route("/api/metrics")
def metrics():

    return jsonify({

        "speed_mae": 5.23,

        "speed_unit": "km/h",

        "acceleration_mae": 0.0354,

        "acceleration_unit": "m/s²",

        "fusion_drift": 8.33,

        "fusion_drift_unit": "%",

        "trajectory_rmse": 46.51,

        "trajectory_rmse_unit": "m",

        "average_position_error": 42.16,

        "average_position_error_unit": "m",

        "maximum_position_error": 81.59,

        "maximum_position_error_unit": "m",

        "evaluation":
            "Selected prototype evaluation segment"

    })


# ============================================================
# CONVERT FUSION DATA TO TRAJECTORY
# ============================================================

def build_trajectory():

    if fusion_data is None:
        return []

    if len(fusion_data) == 0:
        return []

    df = fusion_data.copy()

    time_col = find_any_column(
        df,
        [
            "time",
            "timestamp"
        ]
    )

    east_col = find_any_column(
        df,
        [
            "east",
            "easting"
        ]
    )

    north_col = find_any_column(
        df,
        [
            "north",
            "northing"
        ]
    )

    lat_col = find_any_column(
        df,
        [
            "latitude",
            "lat"
        ]
    )

    lon_col = find_any_column(
        df,
        [
            "longitude",
            "lon"
        ]
    )

    speed_col = find_any_column(
        df,
        [
            "speed",
            "velocity"
        ]
    )

    error_col = find_any_column(
        df,
        [
            "position_error",
            "error"
        ]
    )

    heading_col = find_any_column(
        df,
        [
            "heading",
            "yaw"
        ]
    )

    result = []

    previous_east = None
    previous_north = None

    previous_time = None

    cumulative_distance = 0.0

    for i in range(len(df)):

        row = df.iloc[i]

        time_value = (
            safe_float(row[time_col], i * 0.1)
            if time_col
            else i * 0.1
        )

        east = (
            safe_float(row[east_col])
            if east_col
            else 0.0
        )

        north = (
            safe_float(row[north_col])
            if north_col
            else 0.0
        )

        latitude = (
            safe_float(row[lat_col])
            if lat_col
            else None
        )

        longitude = (
            safe_float(row[lon_col])
            if lon_col
            else None
        )

        heading = (
            safe_float(row[heading_col])
            if heading_col
            else None
        )

        # ----------------------------------------------------
        # Calculate movement between points
        # ----------------------------------------------------

        step_distance = 0.0

        if (
            previous_east is not None
            and previous_north is not None
        ):

            dx = east - previous_east
            dy = north - previous_north

            step_distance = math.sqrt(
                dx * dx +
                dy * dy
            )

            cumulative_distance += step_distance

        # ----------------------------------------------------
        # Estimate speed from trajectory
        # ----------------------------------------------------

        speed = None

        if speed_col:

            speed = safe_float(
                row[speed_col]
            )

        elif previous_time is not None:

            dt = time_value - previous_time

            if dt > 0:

                speed = (
                    step_distance /
                    dt *
                    3.6
                )

        if speed is None:
            speed = 0.0

        # ----------------------------------------------------
        # Position error
        # ----------------------------------------------------

        if error_col:

            position_error = safe_float(
                row[error_col]
            )

        else:

            # If explicit error is unavailable,
            # calculate a smooth relative indicator.
            #
            # This is ONLY for dashboard visualization.
            # It is not used as a benchmark metric.

            position_error = (
                abs(
                    step_distance -
                    (
                        np.mean(
                            [
                                step_distance,
                                1.0
                            ]
                        )
                    )
                )
                * 5
            )

        result.append({

            "index": i,

            "time":
                time_value,

            "east":
                east,

            "north":
                north,

            "latitude":
                latitude,

            "longitude":
                longitude,

            "speed":
                speed,

            "heading":
                heading,

            "step_distance":
                step_distance,

            "distance":
                cumulative_distance,

            "position_error":
                position_error

        })

        previous_east = east
        previous_north = north

        previous_time = time_value

    return result


# ============================================================
# TRAJECTORY API
# ============================================================

@app.route("/api/trajectory")
def trajectory():

    try:

        points = build_trajectory()

        # Keep response manageable
        max_points = 800

        if len(points) > max_points:

            indices = np.linspace(
                0,
                len(points) - 1,
                max_points
            ).astype(int)

            points = [
                points[i]
                for i in indices
            ]

        return jsonify({

            "success": True,

            "source":
                "ai_fusion_results.csv",

            "points":
                points

        })

    except Exception as error:

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error": str(error),

            "points": []

        }), 500


# ============================================================
# LIVE SIMULATION
# ============================================================

@app.route(
    "/api/simulation",
    methods=["GET"]
)
def simulation():

    try:

        points = build_trajectory()

        if len(points) == 0:

            return jsonify({

                "success": False,

                "error":
                    "No fusion trajectory available",

                "points": []

            }), 404

        scenario = request.args.get(
            "scenario",
            "live"
        )

        # ----------------------------------------------------
        # SCENARIO 1
        # LIVE SIMULATION
        # ----------------------------------------------------

        if scenario == "live":

            selected = points

            scenario_name = (
                "LIVE GNSS OUTAGE"
            )

            outage_start_ratio = 0.20
            outage_end_ratio = 0.82

        # ----------------------------------------------------
        # SCENARIO 2
        # ALTERNATE LIVE SCENARIO
        # ----------------------------------------------------

        else:

            total = len(points)

            start = int(
                total * 0.08
            )

            end = int(
                total * 0.92
            )

            selected = points[
                start:end
            ]

            scenario_name = (
                "URBAN CANYON SCENARIO"
            )

            outage_start_ratio = 0.12
            outage_end_ratio = 0.88

        total_points = len(selected)

        outage_start = int(
            total_points *
            outage_start_ratio
        )

        outage_end = int(
            total_points *
            outage_end_ratio
        )

        simulation_points = []

        for i, point in enumerate(selected):

            if i < outage_start:

                phase = "GNSS AVAILABLE"

                gnss = True

                mode = "GNSS + INS"

            elif i < outage_end:

                if i < outage_start + max(
                    1,
                    int(
                        total_points * 0.04
                    )
                ):

                    phase = (
                        "GNSS OUTAGE DETECTED"
                    )

                else:

                    phase = (
                        "AI + IMU DEAD RECKONING"
                    )

                gnss = False

                mode = (
                    "AI + IMU DEAD RECKONING"
                )

            else:

                phase = "GNSS RESTORED"

                gnss = True

                mode = "GNSS + INS RE-FUSION"

            simulation_points.append({

                **point,

                "simulation_index":
                    i,

                "progress":
                    (
                        i /
                        max(
                            1,
                            total_points - 1
                        )
                    ) * 100,

                "phase":
                    phase,

                "gnss":
                    gnss,

                "mode":
                    mode,

                "ai_active":
                    not gnss

            })

        return jsonify({

            "success": True,

            "scenario":
                scenario_name,

            "scenario_type":
                scenario,

            "total_points":
                len(simulation_points),

            "points":
                simulation_points

        })

    except Exception as error:

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ============================================================
# AI SPEED PREDICTION
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        if data is None:

            return jsonify({

                "success": False,

                "error":
                    "No JSON data received"

            }), 400

        features = data.get(
            "features"
        )

        if features is None:

            return jsonify({

                "success": False,

                "error":
                    "Missing 'features' array"

            }), 400

        if (
            speed_model is None
            or speed_scaler is None
        ):

            return jsonify({

                "success": False,

                "error":
                    "AI speed model is not loaded"

            }), 500

        features = np.asarray(
            features,
            dtype=float
        )

        if features.ndim == 1:

            features = features.reshape(
                1,
                -1
            )

        expected_features = (
            speed_scaler.n_features_in_
        )

        if (
            features.shape[1]
            != expected_features
        ):

            return jsonify({

                "success": False,

                "error":
                    (
                        f"Expected "
                        f"{expected_features} "
                        f"features, received "
                        f"{features.shape[1]}"
                    )

            }), 400

        scaled = speed_scaler.transform(
            features
        )

        prediction = (
            speed_model.predict(
                scaled
            )
        )

        speed = max(
            0.0,
            safe_float(
                prediction[0]
            )
        )

        return jsonify({

            "success": True,

            "predicted_speed":
                speed,

            "unit":
                "km/h"

        })

    except Exception as error:

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# ============================================================
# LEGACY SIMULATE API
# ============================================================

@app.route(
    "/api/simulate",
    methods=["POST"]
)
def simulate():

    try:

        data = (
            request.get_json()
            or {}
        )

        phase = data.get(
            "phase",
            "normal"
        )

        if phase == "outage":

            return jsonify({

                "success": True,

                "gnss": False,

                "mode":
                    "AI + IMU DEAD RECKONING",

                "status":
                    "GNSS SIGNAL LOST",

                "message":
                    "Switching to intelligent dead reckoning",

                "ai_active": True

            })

        elif phase == "restore":

            return jsonify({

                "success": True,

                "gnss": True,

                "mode":
                    "GNSS + INS",

                "status":
                    "GNSS RESTORED",

                "message":
                    "Seamless GNSS-INS fusion resumed",

                "ai_active": False

            })

        else:

            return jsonify({

                "success": True,

                "gnss": True,

                "mode":
                    "GNSS + INS",

                "status":
                    "NORMAL",

                "message":
                    "GNSS navigation active",

                "ai_active": False

            })

    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("                 NAV-X BACKEND")
    print("=" * 65)
    print("AI-ML Intelligent Dead Reckoning System")
    print("SIH26168 | ISRO | Smart Vehicles")
    print("=" * 65)
    print("Server: http://127.0.0.1:5000")
    print("=" * 65)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )