import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const API = "http://127.0.0.1:5000";

function App() {
  const [backendOnline, setBackendOnline] = useState(false);
  const [phase, setPhase] = useState("GNSS AVAILABLE");
  const [speed, setSpeed] = useState(42.3);
  const [position, setPosition] = useState({
    east: 0,
    north: 0,
  });
  const [error, setError] = useState(0);
  const [gnss, setGnss] = useState(true);
  const [aiActive, setAiActive] = useState(false);
  const [running, setRunning] = useState(false);

  const [metrics, setMetrics] = useState({
    drift: 8.33,
    rmse: 46.51,
    avgError: 42.16,
    maxError: 81.59,
  });

  const [mouse, setMouse] = useState({
    x: 50,
    y: 50,
  });

  /* =====================================================
     BACKEND CHECK
     ===================================================== */

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch(`${API}/api/health`);

        if (response.ok) {
          setBackendOnline(true);

          try {
            const metricResponse = await fetch(
              `${API}/api/metrics`
            );

            if (metricResponse.ok) {
              const data = await metricResponse.json();

              setMetrics({
                drift: data.fusion_drift ?? 8.33,
                rmse: data.trajectory_rmse ?? 46.51,
                avgError:
                  data.average_position_error ?? 42.16,
                maxError:
                  data.maximum_position_error ?? 81.59,
              });
            }
          } catch {
            console.log("Metrics unavailable");
          }
        } else {
          setBackendOnline(false);
        }
      } catch {
        setBackendOnline(false);
      }
    };

    checkBackend();

    const interval = setInterval(
      checkBackend,
      5000
    );

    return () => clearInterval(interval);
  }, []);

  /* =====================================================
     MOUSE LIGHT
     ===================================================== */

  const handleMouseMove = (e) => {
    setMouse({
      x: (e.clientX / window.innerWidth) * 100,
      y: (e.clientY / window.innerHeight) * 100,
    });
  };

  /* =====================================================
     LIVE SIMULATION
     ===================================================== */

  const runSimulation = async () => {
    if (running) return;

    setRunning(true);

    /* ---------------- NORMAL ---------------- */

    setPhase("GNSS AVAILABLE");
    setGnss(true);
    setAiActive(false);
    setSpeed(42.3);
    setPosition({
      east: 0,
      north: 0,
    });
    setError(0);

    try {
      await fetch(`${API}/api/simulate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          phase: "normal",
        }),
      });
    } catch {}

    await wait(2000);

    /* ---------------- GNSS LOST ---------------- */

    setPhase("GNSS OUTAGE DETECTED");
    setGnss(false);
    setAiActive(true);

    try {
      await fetch(`${API}/api/simulate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          phase: "outage",
        }),
      });
    } catch {}

    await wait(1500);

    /* ---------------- DEAD RECKONING ---------------- */

    setPhase("AI + IMU DEAD RECKONING");

    let east = 0;
    let north = 0;
    let currentSpeed = 42.3;

    for (let i = 0; i < 35; i++) {
      await wait(180);

      currentSpeed =
        30 +
        Math.sin(i / 3) * 8 +
        Math.random() * 3;

      north -= currentSpeed * 0.18;

      east +=
        Math.sin(i / 5) *
        3;

      const currentError =
        Math.sqrt(
          east * east +
            (north + i * 7) *
              (north + i * 7)
        ) * 0.08;

      setSpeed(
        Number(
          currentSpeed.toFixed(1)
        )
      );

      setPosition({
        east: Number(east.toFixed(1)),
        north: Number(
          north.toFixed(1)
        ),
      });

      setError(
        Number(
          Math.abs(currentError).toFixed(1)
        )
      );
    }

    /* ---------------- RESTORE ---------------- */

    setPhase("GNSS RESTORED");
    setGnss(true);
    setAiActive(false);

    try {
      await fetch(`${API}/api/simulate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          phase: "restore",
        }),
      });
    } catch {}

    await wait(1800);

    /* ---------------- NORMAL AGAIN ---------------- */

    setPhase("GNSS AVAILABLE");
    setGnss(true);
    setAiActive(false);
    setError(0);

    setRunning(false);
  };

  return (
    <div
      className="app"
      onMouseMove={handleMouseMove}
      style={{
        "--mouse-x": `${mouse.x}%`,
        "--mouse-y": `${mouse.y}%`,
      }}
    >
      <div className="background-grid"></div>
      <div className="mouse-light"></div>

      {/* =================================================
          NAVBAR
      ================================================= */}

      <header className="navbar">
        <div className="brand">
          <div className="brand-icon">
            <div className="brand-ring"></div>
            <span>NX</span>
          </div>

          <div>
            <div className="brand-name">
              NAV-X
            </div>

            <div className="brand-subtitle">
              INTELLIGENT DEAD RECKONING
            </div>
          </div>
        </div>

        <div className="nav-right">
          <div className="system-tag">
            <span className="pulse-dot"></span>
            SIH26168
          </div>

          <div className="org-tag">
            ISRO · SMART VEHICLES
          </div>
        </div>
      </header>

      <main>

        {/* =================================================
            HERO
        ================================================= */}

        <section className="hero">

          <div className="hero-left">

            <div className="eyebrow">
              <span className="eyebrow-line"></span>
              AI-ML NAVIGATION ENGINE
            </div>

            <h1>
              Seamless navigation
              <br />
              <span>
                when GNSS disappears.
              </span>
            </h1>

            <p className="hero-description">
              An intelligent dead reckoning
              system combining smartphone
              IMU sensors, AI-based motion
              estimation and GNSS fusion to
              maintain continuous vehicle
              positioning during GNSS-denied
              conditions.
            </p>

            <div className="hero-buttons">

              <button
                className="primary-button"
                onClick={runSimulation}
                disabled={running}
              >
                <span className="button-icon">
                  ▶
                </span>

                {running
                  ? "SIMULATION RUNNING..."
                  : "RUN LIVE SIMULATION"}
              </button>

              <div className="engine-status">

                <span
                  className={
                    backendOnline
                      ? "status-light online"
                      : "status-light offline"
                  }
                ></span>

                {backendOnline
                  ? "AI ENGINE ONLINE"
                  : "BACKEND OFFLINE"}

              </div>

            </div>

          </div>

          {/* =================================================
              RADAR
          ================================================= */}

          <div className="hero-visual">

            <div className="radar">

              <div className="radar-ring ring-one"></div>
              <div className="radar-ring ring-two"></div>
              <div className="radar-ring ring-three"></div>

              <div className="radar-cross horizontal"></div>
              <div className="radar-cross vertical"></div>

              <div className="radar-sweep"></div>

              <div
                className="vehicle-marker"
                style={{
                  left: `${
                    50 +
                    Math.sin(
                      position.east / 20
                    ) * 12
                  }%`,
                  top: `${
                    50 +
                    Math.sin(
                      position.north / 30
                    ) * 15
                  }%`,
                }}
              >
                <span></span>
              </div>

              <div className="route route-one"></div>
              <div className="route route-two"></div>

              <div className="radar-label label-top">
                GNSS
              </div>

              <div className="radar-label label-bottom">
                IMU
              </div>

              <div className="radar-label label-right">
                AI
              </div>

            </div>

          </div>

        </section>

        {/* =================================================
            CURRENT STATE
        ================================================= */}

        <section className="state-panel">

          <div className="state-title">
            <span className="section-number">
              01
            </span>

            CURRENT NAVIGATION STATE
          </div>

          <div className="state-content">

            <div className="state-main">

              <div className="state-label">
                ACTIVE PHASE
              </div>

              <div className="state-value">

                <span className="state-dot"></span>

                {phase}

              </div>

              <div className="state-description">
                {gnss
                  ? "GNSS signal available — fusion mode active"
                  : "GNSS unavailable — intelligent dead reckoning active"}
              </div>

            </div>

            <div className="state-flow">

              <div
                className={`flow-node ${
                  gnss ? "active" : ""
                }`}
              >
                <span>GNSS</span>
              </div>

              <div className="flow-line active"></div>

              <div className="flow-node active">
                <span>IMU</span>
              </div>

              <div className="flow-line active"></div>

              <div
                className={`flow-node ${
                  aiActive ? "active" : ""
                }`}
              >
                <span>AI</span>
              </div>

              <div className="flow-line active"></div>

              <div className="flow-node active">
                <span>POSITION</span>
              </div>

            </div>

          </div>

        </section>

        {/* =================================================
            LIVE TELEMETRY
        ================================================= */}

        <section className="telemetry-section">

          <div className="section-heading">

            <div>
              <span className="section-number">
                02
              </span>

              <span className="section-title">
                LIVE TELEMETRY
              </span>
            </div>

            <span className="section-caption">
              REAL-TIME SIMULATION
            </span>

          </div>

          <div className="telemetry-grid">

            <TelemetryCard
              label="AI ESTIMATED SPEED"
              value={speed}
              unit="km/h"
            />

            <TelemetryCard
              label="EAST POSITION"
              value={position.east}
              unit="m"
            />

            <TelemetryCard
              label="NORTH POSITION"
              value={position.north}
              unit="m"
            />

            <TelemetryCard
              label="POSITION ERROR"
              value={error}
              unit="m"
            />

          </div>

        </section>

        {/* =================================================
            PERFORMANCE
        ================================================= */}

        <section className="metrics-section">

          <div className="section-heading">

            <div>
              <span className="section-number">
                03
              </span>

              <span className="section-title">
                PERFORMANCE OVERVIEW
              </span>
            </div>

            <span className="section-caption">
              PROTOTYPE EVALUATION
            </span>

          </div>

          <div className="metrics-grid">

            <MetricCard
              label="POSITION DRIFT"
              value={metrics.drift}
              suffix="%"
              description="Selected evaluation segment"
              accent="green"
            />

            <MetricCard
              label="TRAJECTORY RMSE"
              value={metrics.rmse}
              suffix=" m"
              description="Path reconstruction error"
              accent="blue"
            />

            <MetricCard
              label="AVERAGE ERROR"
              value={metrics.avgError}
              suffix=" m"
              description="Mean position error"
              accent="purple"
            />

            <MetricCard
              label="MAX POSITION ERROR"
              value={metrics.maxError}
              suffix=" m"
              description="Maximum observed error"
              accent="orange"
            />

          </div>

        </section>

        {/* =================================================
            AI MODELS
        ================================================= */}

        <section className="models-section">

          <div className="section-heading">

            <div>
              <span className="section-number">
                04
              </span>

              <span className="section-title">
                AI MOTION MODELS
              </span>
            </div>

            <span className="section-caption">
              EDGE-ORIENTED INFERENCE
            </span>

          </div>

          <div className="models-grid">

            <ModelCard
              icon="V"
              title="AI SPEED ESTIMATION"
              value="5.23"
              unit="km/h MAE"
              text="Estimates vehicle speed from noisy smartphone IMU measurements."
            />

            <ModelCard
              icon="A"
              title="AI ACCELERATION"
              value="0.0354"
              unit="m/s² MAE"
              text="Predicts longitudinal motion to improve inertial propagation."
            />

            <ModelCard
              icon="F"
              title="FUSION ENGINE"
              value="8.33"
              unit="% DRIFT"
              text="Combines AI estimates with inertial navigation and GNSS."
            />

          </div>

        </section>

        {/* =================================================
            WORKFLOW
        ================================================= */}

        <section className="workflow-section">

          <div className="section-heading">

            <div>
              <span className="section-number">
                05
              </span>

              <span className="section-title">
                NAVIGATION WORKFLOW
              </span>
            </div>

          </div>

          <div className="workflow">

            <WorkflowStep
              number="01"
              title="SENSOR INPUT"
              text="Accelerometer · Gyroscope · GNSS"
            />

            <div className="workflow-arrow">
              →
            </div>

            <WorkflowStep
              number="02"
              title="AI PROCESSING"
              text="Speed · Acceleration · Motion features"
            />

            <div className="workflow-arrow">
              →
            </div>

            <WorkflowStep
              number="03"
              title="DEAD RECKONING"
              text="Velocity · Heading · Position"
            />

            <div className="workflow-arrow">
              →
            </div>

            <WorkflowStep
              number="04"
              title="FUSION"
              text="GNSS + IMU + AI"
            />

            <div className="workflow-arrow">
              →
            </div>

            <WorkflowStep
              number="05"
              title="FINAL POSITION"
              text="Continuous navigation output"
            />

          </div>

        </section>

        {/* =================================================
            SCENARIO
        ================================================= */}

        <section className="simulation-section">

          <div className="simulation-card">

            <div className="simulation-left">

              <span className="simulation-label">
                LIVE SCENARIO
              </span>

              <h2>
                GNSS-denied
                <br />
                <span>
                  navigation simulation
                </span>
              </h2>

              <p>
                Simulate a complete GNSS
                outage and observe how the
                AI-assisted inertial system
                maintains navigation continuity
                before returning to GNSS-aided
                positioning.
              </p>

              <button
                className="simulation-button"
                onClick={runSimulation}
                disabled={running}
              >
                {running
                  ? "SCENARIO RUNNING..."
                  : "START SCENARIO"}

                <span>↗</span>
              </button>

            </div>

            <div className="simulation-timeline">

              <TimelineItem
                active={phase === "GNSS AVAILABLE"}
                title="GNSS AVAILABLE"
                description="Normal navigation"
              />

              <TimelineItem
                active={
                  phase ===
                  "GNSS OUTAGE DETECTED"
                }
                title="GNSS LOST"
                description="Outage detected"
              />

              <TimelineItem
                active={
                  phase ===
                  "AI + IMU DEAD RECKONING"
                }
                title="AI + IMU"
                description="Dead reckoning active"
              />

              <TimelineItem
                active={
                  phase ===
                  "GNSS RESTORED"
                }
                title="GNSS RESTORED"
                description="Fusion recovery"
              />

            </div>

          </div>

        </section>

      </main>

      <footer>

        <div>
          <strong>NAV-X</strong>
          <span>
            {" "}
            · Intelligent Dead Reckoning
          </span>
        </div>

        <div>
          SIH 2026 · SIH26168 · ISRO
        </div>

      </footer>

    </div>
  );
}


/* ============================================================
   COMPONENTS
============================================================ */

function TelemetryCard({
  label,
  value,
  unit,
}) {
  return (
    <div className="telemetry-card">

      <div className="telemetry-label">
        {label}
      </div>

      <div className="telemetry-value">
        {Number(value).toFixed(1)}

        <small>
          {unit}
        </small>
      </div>

      <div className="telemetry-status">
        ● LIVE
      </div>

    </div>
  );
}


function MetricCard({
  label,
  value,
  suffix,
  description,
  accent,
}) {
  return (
    <div
      className={`metric-card ${accent}`}
    >

      <div className="card-top">

        <span>{label}</span>

        <span className="metric-symbol">
          ◈
        </span>

      </div>

      <div className="metric-value">

        {value}

        <small>
          {suffix}
        </small>

      </div>

      <div className="metric-description">
        {description}
      </div>

      <div className="metric-line"></div>

    </div>
  );
}


function ModelCard({
  icon,
  title,
  value,
  unit,
  text,
}) {
  return (
    <div className="model-card">

      <div className="model-icon">
        {icon}
      </div>

      <div className="model-content">

        <div className="model-title">
          {title}
        </div>

        <div className="model-result">

          {value}

          <small>
            {unit}
          </small>

        </div>

        <p>
          {text}
        </p>

      </div>

    </div>
  );
}


function WorkflowStep({
  number,
  title,
  text,
}) {
  return (
    <div className="workflow-step">

      <div className="workflow-number">
        {number}
      </div>

      <div className="workflow-title">
        {title}
      </div>

      <div className="workflow-text">
        {text}
      </div>

    </div>
  );
}


function TimelineItem({
  active,
  title,
  description,
}) {
  return (
    <div
      className={`timeline-item ${
        active ? "active" : ""
      }`}
    >

      <div className="timeline-dot"></div>

      <div>

        <div className="timeline-title">
          {title}
        </div>

        <div className="timeline-description">
          {description}
        </div>

      </div>

    </div>
  );
}


/* ============================================================
   HELPER
============================================================ */

function wait(ms) {
  return new Promise((resolve) =>
    setTimeout(resolve, ms)
  );
}


/* ============================================================
   START REACT
============================================================ */

createRoot(
  document.getElementById("root")
).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);