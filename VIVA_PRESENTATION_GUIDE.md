# 🎓 Adaptive AI-Driven CPU Scheduling: Team Presentation & Viva Defense Guide

> **Project Title:** Adaptive AI-Driven CPU Scheduling: Dynamic Workload Adaptation & Real-Time Telemetry in Multi-Core Operating Systems  
> **Course:** Operating Systems  
> **Team Members:**  
> - **Jyotirmoy Bhunia** (`24BCE0679`) — *Predictive Modelling & Dynamic Scheduler Policy Design*  
> - **Abhiram Malepati** (`24BCE2089`) — *Real-Time Telemetry Pipeline & Workload Classification*  
> - **Raghav G** (`24BCE0676`) — *Fairness, Starvation Control & Safe Fallbacks*

---

## ⏱️ Recommended 6-Minute Presentation Flow

```
[ Team Introduction & Abstract ] ──────────► (30 sec — Joint)
         │
         ▼
[ Member 1: Jyotirmoy Bhunia ] ────────────► (1.5 min — ML Classifier & Policy Switching)
         │
         ▼
[ Member 2: Abhiram Malepati ] ────────────► (1.5 min — Telemetry Pipeline & Feature Vector)
         │
         ▼
[ Member 3: Raghav G ] ────────────────────► (1.5 min — Starvation Guard, Aging & Dual-Core)
         │
         ▼
[ Live Interactive Dashboard Demo & Q&A ] ──► (1.5 min — Presets 1, 2, 3 + Faculty Q&A)
```

---

## 👤 Member 1: Jyotirmoy Bhunia (`24BCE0679`)
**Focus Area:** Predictive Modelling & Dynamic Scheduler Policy Design  
**Owned Code:** `ml/data_generator.py`, `ml/train_model.py`, and Policy Selection logic in `scheduler/adaptive_scheduler.py`.

### 🎤 Presentation Speech
> *"Respected faculty, I was responsible for the **Predictive Machine Learning Engine and Dynamic Policy Switching**.*
>
> *Traditional operating systems rely on static scheduling algorithms like FCFS or Round Robin. While simple, they cannot adapt when workload profiles fluctuate between heavy CPU rendering and interactive I/O.*
>
> *I designed our offline machine learning pipeline using a **Random Forest Classifier** trained on 1,500 balanced workload samples across three target classes: **CPU-bound, I/O-bound, and Mixed**.*
>
> *At runtime, our system does not perform slow in-kernel retraining. Instead, it runs instantaneous inference in under 2 milliseconds. Guided by the workload diagnosis, our policy engine dynamically switches algorithms:*
> * *For **CPU-bound batches with high variance**, it engages **SRTF** to eliminate the Convoy Effect.*
> * *For **high-concurrency mixed batches**, it applies **Dynamic Round Robin** with adaptive time quanta.*
> * *For **urgent tasks**, it preempts with **Priority Scheduling**.*
>
> *In Dual-Core mode, this engine serves as an asymmetric dispatcher: routing heavy compute tasks to Core 0 while reserving Core 1 for low-latency interactive bursts."*

### 🖥️ Dashboard Demo Steps
1. **Show the ML Prediction Banner:** Point out the top card showing the predicted class (e.g., `CPU-bound [Confidence: 96%]`).
2. **Preset 1 (Convoy Effect Demo):** Show how the policy switches from FCFS to SRTF to finish the 2-tick processes ($P_2$–$P_5$) before letting the 30-tick process ($P_1$) hog the CPU.
3. **Preset 3 (Mixed Batch & Interactive):** Toggle between Single Core and Dual Core to demonstrate how the policy routes compute tasks to Core 0 and interactive tasks to Core 1.

### ❓ Top Viva Questions & Answers
* **Q: Why Random Forest instead of Deep Learning / Neural Networks?**  
  * **Answer:** *"In OS kernel scheduling, decision overhead must be sub-millisecond. Deep neural networks require expensive matrix multiplications and GPU/TPU acceleration. A Random Forest with 100 decision trees executes inference on CPU in under 2 milliseconds with $>98\%$ accuracy and zero floating-point training lag."*
* **Q: Does the model retrain when the user enters processes in the table?**  
  * **Answer:** *"No, sir. Training is an offline stage that created `model.joblib`. Live table inputs undergo instantaneous feature extraction and inference only."*

---

## 👤 Member 2: Abhiram Malepati (`24BCE2089`)
**Focus Area:** Real-Time Telemetry Pipeline & Workload Classification  
**Owned Code:** `ml/predictor.py`, `metrics/calculator.py`, and Telemetry Logging in `dashboard.py`.

### 🎤 Presentation Speech
> *"Respected faculty, I engineered the **Real-Time Telemetry Pipeline and Workload Feature Instrumentation**.*
>
> *An adaptive operating system cannot make intelligent decisions in the dark. My role was to provide the real-time 'sensors' that inspect incoming workloads and feed metrics back into the scheduling loop.*
>
> *Before execution, my module extracts a **5-dimensional Predictive Feature Vector** directly from process metadata:*
> 1. *`avg_burst_time` — average CPU execution demand.*
> 2. *`burst_variance` — detects heterogeneous task mixtures (the primary trigger for the Convoy Effect).*
> 3. *`cpu_usage_pct` — estimated processor utilization.*
> 4. *`io_burst_ratio` — ratio of I/O wait cycles relative to total execution.*
> 5. *`process_count` — instantaneous ready queue pressure.*
>
> *During runtime, telemetry monitors tick-by-tick performance: tracking `last_cpu_time` to calculate continuous process wait times, recording context switches, and monitoring per-core utilization percentages.*
>
> *Finally, every single scheduling switch is instrumented into an **Adaptation Audit Log** that explains not just what was scheduled, but the exact telemetry reason why."*

### 🖥️ Dashboard Demo Steps
1. **Feature Vector Expander:** Expand the *"Predictive Feature Vector"* box on the dashboard to show the 5 extracted values computed from the active process list.
2. **Adaptation Telemetry Log:** Scroll through the live table showing `Timestamp | Core | Policy | Queue Size | Active PIDs | Telemetry & Decision Reason`.
3. **Telemetry Metrics Cards:** Highlight Average Waiting Time (AWT), Turnaround Time (ATAT), Response Time (ART), Throughput, and Core 0 / Core 1 Utilization gauges.

### ❓ Top Viva Questions & Answers
* **Q: Why are these specific 5 features chosen for the vector?**  
  * **Answer:** *"These 5 features capture the fundamental dimensions of OS workload behavior. Mean burst and variance indicate whether tasks are uniform or convoy-prone. I/O ratio and CPU percentage distinguish computation-heavy jobs from interactive tasks. Process count reflects queue depth and concurrency."*
* **Q: How is per-core utilization calculated?**  
  * **Answer:** *"Telemetry tracks active execution intervals on each core divided by the total makespan: $\text{Util} = \frac{\sum \text{busy ticks}}{\text{total makespan}} \times 100\%$. In our dual-core benchmarks, it demonstrates $>90\%$ balanced utilization."*

---

## 👤 Member 3: Raghav G (`24BCE0676`)
**Focus Area:** Fairness, Starvation Control & Safe Fallbacks  
**Owned Code:** `core/process.py`, Anti-Starvation Watchdog & Work-Stealing in `scheduler/adaptive_scheduler.py`, and `tests/test_schedulers.py`.

### 🎤 Presentation Speech
> *"Respected faculty, my responsibility was ensuring **Operating System Stability, Fairness, and Anti-Starvation Guarantees**.*
>
> *While ML and shortest-job-first policies optimize throughput and waiting time, pure optimization introduces a fatal flaw in operating systems: **Starvation of long or low-priority threads**.*
>
> *To solve this, I engineered a dual-layer safety mechanism:*
> 1. * **Continuous Priority Aging:** Every thread waiting in the ready queue without CPU access has its effective priority dynamically promoted: $\text{prio} = \max(1, \text{base\_prio} - \lfloor \frac{\text{wait}}{\text{interval}} \rfloor)$. This ensures low-priority background jobs gradually rise to VIP status.*
> 2. * **Rule 0 Deterministic Fallback Watchdog:** Telemetry actively checks the continuous wait duration of every process. If any process sits starved beyond a configured threshold (e.g., 15 ticks), the scheduler **overrides all ML and priority decisions**, issuing an emergency preemption that rescues the starved process immediately.*
>
> *In our Dual-Core implementation, I also designed the **Work-Stealing algorithm**: if Core 1 finishes its interactive queue and goes idle, it safely steals waiting jobs from Core 0 to prevent CPU cycle wastage."*

### 🖥️ Dashboard Demo Steps
1. **Preset 2 (Starvation Risk Scenario):** Run this preset and point directly to the **Red "Starvation Guard" row** at $t=17$ in the Adaptation Log and the Red slice on the Gantt chart.
2. **Starvation Rescues Metric Card:** Show the counter reading `1 Starvation Rescue`.
3. **Explain the SRTF Trade-Off:** Show why SRTF has an AWT of $5.40$ while Adaptive has $6.20$: explain that the $0.8$-tick difference is the deliberate **fairness tax** paid to prevent $P_1$ from hanging indefinitely.
4. **Work-Stealing in Dual Core:** Show how Core 1 steals compute tasks when interactive tasks finish, bringing Core 1 utilization from 0% up to ~85%.

### ❓ Top Viva Questions & Answers
* **Q: Why does your Adaptive Single-Core have slightly higher AWT than SRTF in Preset 2?**  
  * **Answer:** *"SRTF is mathematically the optimal algorithm for AWT, but it achieves this by completely starving long jobs. Pure SRTF would hold $P_1$ indefinitely if short jobs kept arriving. Our scheduler's Anti-Starvation Watchdog intervened at $t=17$ to rescue $P_1$. Sacrificing just $0.8$ ticks of average waiting time is an essential engineering trade-off to guarantee starvation freedom."*
* **Q: How does Work-Stealing operate across cores?**  
  * **Answer:** *"When Core 1 finishes its interactive tasks and finds its ready queue empty, it inspects Core 0's backlog and steals the shortest remaining process. This keeps both cores utilized without stalling compute tasks."*

---

## 🎯 Master Viva Defense Cheat-Sheet

| Topic / Question | What Faculty Might Say | The Winning Defense |
| :--- | :--- | :--- |
| **Synthetic Dataset** | *"Why didn't you use real system traces?"* | Real OS traces are $95\%$ idle system daemons, causing massive class imbalance where ML predicts 'I/O-bound' blindly. Synthetic modeling provides balanced classes ($500$ each) and allows deterministic stress testing of edge cases (Convoy Effect, Starvation). |
| **SRTF vs Adaptive AWT** | *"Why is SRTF's AWT ($5.40$) better than Adaptive ($6.20$)?* | SRTF minimizes AWT by committing the cardinal sin of operating systems: indefinite starvation of $P_1$. Adaptive intentionally pays a $0.8$-tick 'fairness tax' to rescue $P_1$ and guarantee system liveness. |
| **Model Invalidation** | *"Do we need to retrain if we change scheduling rules?"* | No. The ML classifier only diagnoses the workload profile (CPU/IO/Mixed). The scheduling algorithms and safety watchdogs operate independently downstream. |
| **Multi-Core Overhead** | *"Doesn't Dual-Core add context switch overhead?"* | Work-stealing only activates when a core is completely idle, avoiding unnecessary task migration. Parallel throughput easily offsets minor migration costs. |
