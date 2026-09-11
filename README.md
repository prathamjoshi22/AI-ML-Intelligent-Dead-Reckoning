# AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation

## SIH 2026 — SIH26168

An AI-ML based vehicle positioning system designed to maintain continuous navigation during GNSS outages using smartphone IMU sensors, machine learning and dead reckoning.

## Problem

GNSS signals can become unavailable in tunnels, underground parking areas, dense urban environments, forests, valleys, or due to signal interference and jamming.

When GNSS is lost, conventional navigation systems may experience positioning interruptions.

## Proposed Solution

Our system combines:

* Smartphone IMU sensors
* AI-based speed estimation
* AI-based acceleration estimation
* Inertial dead reckoning
* GNSS/IMU fusion
* GNSS outage detection and recovery
* Flask backend
* React-based navigation dashboard

The system switches from GNSS-assisted positioning to inertial dead reckoning during a GNSS outage and returns to GNSS-assisted positioning when the signal becomes available again.

## System Workflow

```text
Smartphone Sensors
       ↓
Signal Processing
       ↓
Sensor Features
       ↓
AI Motion Estimation
       ↓
Dead Reckoning
       ↓
GNSS + IMU Fusion
       ↓
Estimated Vehicle Position
       ↓
Navigation Dashboard
```

## AI/ML Models

### 1. AI Speed Estimation

The speed model uses smartphone sensor features extracted from:

* Accelerometer
* Gravity-compensated acceleration
* Gyroscope
* Sensor magnitudes
* Recent motion statistics

The model estimates vehicle speed during GNSS-denied conditions.

**Model:** MLP Regressor

**Test MAE:** 5.23 km/h

### 2. AI Acceleration Estimation

The acceleration model estimates vehicle acceleration from smartphone sensor features.

**Model:** MLP Regressor

**Test MAE:** 0.0354 m/s²

## Dead Reckoning

During a GNSS outage, the system uses the last known position and estimates subsequent vehicle movement using inertial sensor measurements and AI-based motion estimation.

The estimated trajectory is continuously updated until GNSS becomes available again.

## GNSS Mode Switching

```text
GNSS Available
      ↓
GNSS + IMU Fusion
      ↓
GNSS Signal Lost
      ↓
AI + IMU Dead Reckoning
      ↓
GNSS Signal Restored
      ↓
GNSS + IMU Fusion
```

This allows the system to maintain positioning continuity during temporary GNSS outages.

## Technology Stack

### AI / Navigation

* Python
* NumPy
* Pandas
* SciPy
* Scikit-learn
* Joblib

### Backend

* Python
* Flask
* REST APIs

### Frontend

* React.js
* Vite
* JavaScript
* CSS

### Dataset

The prototype was developed and evaluated using the IO-VNBD synchronized smartphone and vehicle dataset.

## Evaluation

The system was evaluated by simulating GNSS outage periods and comparing the estimated trajectory against vehicle ground-truth data.

Evaluation metrics include:

* Final position error
* Drift percentage
* Trajectory RMSE
* Average position error
* Maximum position error

On the selected evaluation segment:

* Ground-truth travelled distance: **979.88 m**
* Final position error: **81.59 m**
* Position drift: **8.33%**
* Trajectory RMSE: **46.51 m**
* Average position error: **42.16 m**
* Maximum position error: **81.59 m**

The 8.33% value represents the drift measured on the selected evaluation segment and should not be interpreted as universal system accuracy.

## Web Application

The project contains a React frontend and Flask backend.

### Frontend

The dashboard displays:

* GNSS status
* Navigation mode
* AI motion information
* Vehicle trajectory
* System metrics
* GNSS outage simulation

### Backend

The Flask backend provides REST APIs connecting the dashboard with the AI/navigation components.

## Project Structure

```text
AI-ML-Intelligent-Dead-Reckoning/
│
├── backend/
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── package.json
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       └── style.css
│
├── src/
│   ├── train_ai_speed_model.py
│   ├── train_ai_acceleration_model.py
│   ├── ai_fusion_dead_reckoning.py
│   ├── evaluate_fusion.py
│   └── other navigation and analysis scripts
│
├── models/
│   └── trained ML models
│
├── .gitignore
└── README.md
```

## Running the Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Backend:

```text
http://127.0.0.1:5000
```

Health check:

```text
http://127.0.0.1:5000/api/health
```

## Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite development server will provide the local frontend URL.

## Future Improvements

* Android smartphone deployment
* OpenStreetMap-based map matching
* Non-Holonomic Constraints (NHC)
* EKF/UKF based sensor fusion
* Improved phone-to-vehicle orientation estimation
* Adaptive vibration filtering
* Real-world vehicle testing
* External IMU support
* Longer GNSS outage evaluation

## Project Status

**Prototype / SIH 2026**

The current implementation demonstrates AI-assisted vehicle positioning and dead reckoning during simulated GNSS outages. Further work is required for robust real-world deployment across different phones, mounting orientations and driving conditions.

## License

This project is developed as an academic/hackathon prototype for SIH 2026.
