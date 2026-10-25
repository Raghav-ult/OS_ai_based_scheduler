# Adaptive AI-Driven CPU Scheduling

> **Dynamic Workload Adaptation & Real-Time Telemetry in Multi-Core Operating Systems**

A production-ready, modular Operating Systems & Machine Learning simulation environment. Combines **Machine Learning workload classification**, **Dual-Core (SMP) symmetric parallel dispatching**, and an **Anti-Starvation Deterministic Fallback Watchdog** with an interactive dark-glassmorphism Streamlit dashboard.

---

## 👥 Project Team & Focus Areas

| Member | Registration No. | Project Focus Area |
|---|---|---|
| **Jyotirmoy Bhunia** | `24BCE0679` | Predictive modelling & dynamic scheduler policy design |
| **Abhiram Malepati** | `24BCE2089` | Real-time telemetry pipeline & workload classification |
| **Raghav G** | `24BCE0676` | Fairness, starvation control & safe fallbacks |

---

## ✨ Key Architecture & Features

| Feature | Details |
|---|---|
| **CPU Architectures** | **⚡ Dual-Core (SMP)** parallel execution lanes + **🖥️ Single-Core** execution |
| **Anti-Starvation Watchdog** | Active telemetry watchdog: forces deterministic fallback preemption when waiting time exceeds threshold |
| **Dynamic Priority Aging** | Continuous priority promotion for threads waiting in ready queues ($\text{prio} = \max(1, \text{prio} - \text{wait} // \text{interval})$) |
| **Classic Schedulers** | FCFS, SJF (Non-preemptive), SRTF (Preemptive SJF), Round Robin, Priority (Static), Priority (with Aging) |
| **Adaptive AI Policies** | Event-driven transitions: SRTF for compute jobs, Dynamic Round Robin for I/O jobs, Priority for real-time bursts |
| **Telemetry Pipeline** | Queue depth, burst variance, remaining times, I/O frequencies, per-core CPU utilization |
| **Interactive UI** | Streamlit dark-glassmorphism dashboard with parallel multi-core Plotly Gantt charts & policy logs |

---

## 🚀 Quickstart

### 1 — Navigate to the project directory
```powershell
cd C:\Users\ADMIN\Desktop\OS_Project
```

### 2 — Activate the virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies
```powershell
pip install -r requirements.txt
```

### 4 — Launch the Interactive Dashboard
```powershell
streamlit run dashboard.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🧪 Automated Benchmarking & Testing

Run Monte Carlo simulation benchmarks across 30+ random workload distributions:
```powershell
python -m scripts.run_benchmarks
```
*(Saves detailed statistical metrics to `data/benchmark_evaluation_summary.csv`)*

Run unit tests:
```powershell
pytest tests/
```

---

## 📁 Repository Structure

```
OS_Project/
├── core/
│   ├── process.py              # Process model, priority aging, interval tracking
│   ├── schedulers.py           # FCFS, SJF, SRTF, RR, Priority Static & Aging
│   └── simulation_engine.py   # Simulation orchestrator across algorithms
├── ml/
│   ├── data_generator.py       # Balanced synthetic workload dataset generator (1,500 samples)
│   ├── train_model.py          # Random Forest training pipeline & persistence
│   └── predictor.py            # Feature extraction & live inference
├── scheduler/
│   └── adaptive_scheduler.py  # Single-Core TrueAdaptive & Dual-Core SMP Dispatcher
├── metrics/
│   └── calculator.py           # AWT, ATAT, ART, Throughput, and DataFrame builders
├── scripts/
│   └── run_benchmarks.py       # Monte Carlo empirical benchmark runner
├── tests/
│   └── test_schedulers.py      # Unit test suite
├── dashboard.py                # Streamlit glassmorphism web UI
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📄 License
MIT License — Academic and educational research.
