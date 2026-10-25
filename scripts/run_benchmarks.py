"""
scripts/run_benchmarks.py
-------------------------
Automated Empirical Evaluation Suite for Adaptive AI CPU Scheduling.

Runs Monte Carlo evaluations across multiple workload configurations,
benchmarking the Adaptive Schedulers against static baselines (FCFS, SJF, RR, Priority).
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.process import Process
from core.simulation_engine import SimulationEngine, generate_processes
from scheduler.adaptive_scheduler import TrueAdaptiveScheduler, DualCoreAdaptiveScheduler


def run_monte_carlo_benchmarks(num_runs: int = 50, process_count: int = 12) -> pd.DataFrame:
    """
    Simulates `num_runs` diverse synthetic workload batches across all scheduling algorithms.
    """
    print(f"\n{'='*75}")
    print(f"  RUNNING EMPIRICAL BENCHMARKS ({num_runs} Runs x {process_count} Processes)")
    print(f"{'='*75}\n")

    results_accumulator = {
        "FCFS":                   {"awt": [], "atat": [], "art": [], "cs": [], "starved": []},
        "SJF (Non-Preemptive)":   {"awt": [], "atat": [], "art": [], "cs": [], "starved": []},
        "SJF (Preemptive / SRTF)":{"awt": [], "atat": [], "art": [], "cs": [], "starved": []},
        "Round Robin (Q=4)":      {"awt": [], "atat": [], "art": [], "cs": [], "starved": []},
        "Priority (Static)":      {"awt": [], "atat": [], "art": [], "cs": [], "starved": []},
        "Priority (with Aging)":  {"awt": [], "atat": [], "art": [], "cs": [], "starved": []},
        "Adaptive Single-Core":   {"awt": [], "atat": [], "art": [], "cs": [], "starved": []},
        "Dual-Core Adaptive SMP": {"awt": [], "atat": [], "art": [], "cs": [], "starved": []},
    }

    start_time = time.time()

    for seed in range(1, num_runs + 1):
        procs = generate_processes(count=process_count, seed=seed)

        # 1. Classic Schedulers
        engine = SimulationEngine(procs)
        classic_results = engine.run_all(time_quantum=4, aging_interval=4)

        # 2. True Adaptive Single-Core
        single_sched = TrueAdaptiveScheduler(base_quantum=4, starvation_threshold=15, enable_aging=True)
        single_res = single_sched.run(procs)

        # 3. Dual-Core Adaptive SMP
        dual_sched = DualCoreAdaptiveScheduler(base_quantum=4, starvation_threshold=15, enable_aging=True)
        dual_res = dual_sched.run(procs)

        all_runs = {
            "FCFS": classic_results["FCFS"],
            "SJF (Non-Preemptive)": classic_results["SJF"],
            "SJF (Preemptive / SRTF)": classic_results["SRTF"],
            "Round Robin (Q=4)": classic_results["RR"],
            "Priority (Static)": classic_results["Priority"],
            "Priority (with Aging)": classic_results["PriorityAging"],
            "Adaptive Single-Core": single_res,
            "Dual-Core Adaptive SMP": dual_res,
        }

        for algo_name, res in all_runs.items():
            awt = res["avg_waiting_time"]
            atat = res["avg_turnaround_time"]
            art = res["avg_response_time"]
            cs = res["context_switches"]

            starved_count = sum(1 for p in res["processes"] if p.waiting_time > 20)

            results_accumulator[algo_name]["awt"].append(awt)
            results_accumulator[algo_name]["atat"].append(atat)
            results_accumulator[algo_name]["art"].append(art)
            results_accumulator[algo_name]["cs"].append(cs)
            results_accumulator[algo_name]["starved"].append(starved_count)

    elapsed = time.time() - start_time

    summary_rows = []
    for algo, data in results_accumulator.items():
        summary_rows.append({
            "Algorithm": algo,
            "Mean AWT": round(float(np.mean(data["awt"])), 2),
            "p95 AWT": round(float(np.percentile(data["awt"], 95)), 2),
            "Mean ATAT": round(float(np.mean(data["atat"])), 2),
            "Mean ART": round(float(np.mean(data["art"])), 2),
            "Avg CSwitches": round(float(np.mean(data["cs"])), 1),
            "Starvation Rate": f"{np.mean(data['starved']):.2f} procs/run",
        })

    summary_df = pd.DataFrame(summary_rows)

    print(summary_df.to_string(index=False))
    print(f"\n{'-'*75}")
    print(f"  Benchmark complete in {elapsed:.2f} seconds.")

    out_csv = os.path.join(_ROOT, "data", "benchmark_evaluation_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"  Saved benchmark metrics to -> {out_csv}")
    print(f"{'='*75}\n")

    return summary_df


if __name__ == "__main__":
    run_monte_carlo_benchmarks(num_runs=30, process_count=10)
