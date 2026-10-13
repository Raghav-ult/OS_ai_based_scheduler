# Adaptive AI-Driven CPU Scheduling

A production-ready, modular Python simulation of AI-driven CPU scheduling. A Machine Learning model classifies incoming workload batches as **CPU-bound**, **I/O-bound**, or **Mixed**, and adaptively selects the optimal scheduling algorithm — all visualised in an interactive Streamlit dashboard.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Classic Schedulers** | FCFS, SJF (preemptive & non-preemptive), Round Robin, Priority |
| **ML Classifier** | Random Forest (200 trees) trained on 1,500 synthetic samples |
| **Adaptive Policy** | CPU-bound → SJF · I/O-bound → RR (dynamic Q) · Mixed → Priority/RR |
| **Dashboard** | Dark glassmorphism UI with Plotly Gantt + comparison charts |
| **Metrics** | AWT, ATAT, ART, Throughput, CPU Utilisation, Context Switches |

---

## 🚀 Quickstart

### 1 — Clone / navigate to the project

```powershell
cd adaptive-cpu-scheduler
```

### 2 — Create and activate a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

### 4 — Generate training data & train the ML model

```powershell
# From the adaptive-cpu-scheduler/ directory:
python -m ml.data_generator      # Creates data/workload_dataset.csv
python -m ml.train_model         # Trains model, prints report, saves ml/model.joblib
```

Expected output:
```
[DataGenerator] Dataset saved → …/data/workload_dataset.csv
[DataGenerator] Shape: (1500, 6)
[Trainer] Loaded dataset: 1500 rows, 6 columns
[Trainer] Train set: 1200  |  Test set: 300
...
  Test Accuracy : 97.33%
  CV Accuracy   : 96.42% ± 0.58%
[Trainer] Model saved → …/ml/model.joblib
```

### 5 — Launch the interactive dashboard

```powershell
streamlit run dashboard.py
```

The dashboard opens at **http://localhost:8501** by default.

---

## 📁 Project Structure

```
adaptive-cpu-scheduler/
├── core/
│   ├── process.py              # Process class with all scheduling attributes
│   ├── schedulers.py           # FCFS, SJF, Round Robin, Priority implementations
│   └── simulation_engine.py   # Process factory + multi-scheduler orchestrator
├── ml/
│   ├── data_generator.py       # Synthetic workload dataset generator (1,500+ rows)
│   ├── train_model.py          # RF training pipeline, evaluation, model persistence
│   └── predictor.py            # Feature extraction + real-time inference
├── scheduler/
│   └── adaptive_scheduler.py  # AI decision layer: ML → algorithm selection
├── metrics/
│   └── calculator.py           # AWT, ATAT, ART, Throughput, DataFrame builders
├── data/                       # Auto-created; stores workload_dataset.csv
├── dashboard.py                # Streamlit web UI
├── requirements.txt
└── README.md
```

---

## 🧪 Running Individual Modules

```powershell
# Test the predictor on a random batch
python -m ml.predictor

# Run a quick simulation via the engine
python -c "
from core.simulation_engine import SimulationEngine, generate_processes
procs = generate_processes(10, seed=42)
engine = SimulationEngine(procs)
results = engine.run_all(time_quantum=4)
for k, r in results.items():
    print(f'{k}: AWT={r[\"avg_waiting_time\"]}, ATAT={r[\"avg_turnaround_time\"]}')
"
```

---

## 🤖 ML Adaptive Policy Mapping

| Predicted Class | Algorithm | Rationale |
|---|---|---|
| **CPU-bound** (0) | SJF (Non-Preemptive) | Minimises turnaround time for long, predictable bursts |
| **I/O-bound** (1) | Round Robin (small dynamic Q) | Ensures low response time; prevents I/O process starvation |
| **Mixed** (2) | Priority / Dynamic RR | Serves urgent processes first; Q tuned to burst variance |

The model uses **5 features** extracted from the live process batch:

- `avg_burst_time` — mean CPU burst across all processes  
- `burst_variance` — variance of burst times  
- `cpu_usage_pct` — estimated CPU utilisation  
- `io_burst_ratio` — mean I/O frequency  
- `process_count` — batch size  

---

## 📊 Dashboard Walkthrough

1. **Sidebar** — Adjust process count, burst range, I/O probability, RR quantum.
2. **Run Button** — Triggers ML inference + all scheduler simulations.
3. **AI Cards** — Workload class, confidence %, selected policy.
4. **Gantt Chart** — Interactive Plotly timeline; tab between AI and classic views.
5. **Process Table** — Per-process AWT / ATAT / ART with heatmap highlights.
6. **Comparison Chart** — Side-by-side bar chart of all 5 schedulers.
7. **Feature Vector** — Bar chart showing the 5 ML features extracted from the batch.
8. **Retrain Button** — Regenerates dataset and retrains model in-browser.

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `SAMPLES_PER_CLASS` | 500 | Rows per class in training data |
| `n_estimators` | 200 | Random Forest trees |
| `test_size` | 0.20 | Train/test split fraction |
| `base_quantum` | 4 | Default RR time quantum |

---

## 📦 Dependencies

```
streamlit>=1.32.0
scikit-learn>=1.4.0
pandas>=2.1.0
numpy>=1.26.0
plotly>=5.20.0
joblib>=1.3.2
```

---

## 📄 License

MIT — free to use and modify for educational and research purposes.
