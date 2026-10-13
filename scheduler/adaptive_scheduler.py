"""
scheduler/adaptive_scheduler.py
---------------------------------
AI Decision Layer — maps ML workload predictions to the optimal
CPU scheduling algorithm and executes it.

Adaptive Policy Mapping (driven by ML model output, not hard rules):
    • CPU-bound  → Shortest Job First (Non-Preemptive)
                   Rationale: SJF minimises average turnaround time when bursts
                   are long and predictable.
    • I/O-bound  → Round Robin with a short, dynamically tuned quantum
                   Rationale: Frequent context-switches keep I/O processes
                   responsive; low waiting time despite short bursts.
    • Mixed      → Priority Scheduling or Dynamic Round Robin
                   Rationale: Heterogeneous workloads need urgency-aware
                   scheduling; quantum is tuned to burst variance.

All decisions are logged with full justification text.
"""

import math
import logging
from typing import List, Dict, Any, Optional

from core.process import Process
from core.schedulers import (
    sjf_non_preemptive,
    round_robin,
    priority_scheduling,
)
from ml.predictor import predict_workload

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AdaptiveScheduler] %(levelname)s – %(message)s",
)


# ---------------------------------------------------------------------------
# Quantum tuning helper
# ---------------------------------------------------------------------------

def _compute_optimal_quantum(processes: List[Process], base_quantum: int = 4) -> int:
    """
    Compute a dynamic Round Robin time quantum based on burst statistics.

    The heuristic targets a quantum ≈ 80th-percentile burst time divided by 4,
    bounded to [2, 20].  This keeps context switches low for long bursts
    while maintaining responsiveness for short ones.

    Args:
        processes   : Batch of processes in the simulation.
        base_quantum: Fallback quantum if statistics can't be computed.

    Returns:
        Integer time quantum (ticks).
    """
    if not processes:
        return base_quantum

    bursts = sorted(p.burst_time for p in processes)
    n = len(bursts)
    p80_index = min(int(math.ceil(0.80 * n)) - 1, n - 1)
    p80_burst = bursts[p80_index]

    quantum = max(2, min(20, p80_burst // 4 or 2))
    return quantum


# ---------------------------------------------------------------------------
# Adaptive scheduler
# ---------------------------------------------------------------------------

class AdaptiveScheduler:
    """
    AI-driven CPU scheduler that:
      1. Uses the trained ML model to classify the workload batch.
      2. Maps the predicted class to the most appropriate algorithm.
      3. Executes the selected algorithm and returns full simulation results.
    """

    CLASS_ALGORITHM_MAP = {
        0: "SJF",           # CPU-bound
        1: "RR",            # I/O-bound
        2: "Priority",      # Mixed (may fall back to Dynamic RR)
    }

    JUSTIFICATIONS = {
        0: (
            "CPU-bound workload detected: processes have long, predictable burst times "
            "and minimal I/O. Shortest Job First (SJF) is selected to minimise average "
            "turnaround time by executing the shortest remaining job first, reducing "
            "convoy effects."
        ),
        1: (
            "I/O-bound workload detected: processes have short CPU bursts interspersed "
            "with frequent I/O waits. Round Robin is selected with a dynamically tuned "
            "time quantum to ensure low response times and fair CPU sharing, preventing "
            "any process from monopolising the CPU during I/O waits."
        ),
        2: (
            "Mixed workload detected: high variance in burst times indicates a blend of "
            "CPU-intensive and I/O-intensive processes. Priority Scheduling is selected "
            "to service urgent (high-priority) processes first while maintaining "
            "throughput for longer jobs. A Dynamic Round Robin fallback is used if "
            "priority values are uniform."
        ),
    }

    def __init__(self, base_quantum: int = 4):
        """
        Args:
            base_quantum: Default Round Robin time quantum. Adaptive logic may override.
        """
        self.base_quantum = base_quantum

    def run(
        self,
        processes: List[Process],
        override_quantum: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full adaptive scheduling pipeline.

        Steps:
            1. Call ML predictor to classify the workload.
            2. Select the algorithm based on predicted class.
            3. Optionally tune the RR quantum dynamically.
            4. Execute the chosen scheduler.
            5. Return combined prediction metadata + scheduling results.

        Args:
            processes       : Batch of Process objects to schedule.
            override_quantum: If provided, use this fixed quantum instead of dynamic.

        Returns:
            Dictionary containing:
                'prediction'         – full output from predict_workload()
                'selected_algorithm' – human-readable algorithm name
                'justification'      – detailed reasoning text
                'time_quantum'       – quantum used (for RR variants)
                'processes'          – list of Process with filled metrics
                'timeline'           – Gantt execution intervals
                'context_switches'   – int
                'avg_waiting_time'   – float
                'avg_turnaround_time'– float
                'avg_response_time'  – float
                'throughput'         – float
        """
        if not processes:
            raise ValueError("Cannot schedule an empty process list.")

        # ── Step 1: ML classification ──────────────────────────────────────
        prediction = predict_workload(processes)
        class_id = prediction["class_id"]
        label = prediction["label"]
        confidence = prediction["confidence"]

        logger.info(
            "ML Prediction → %s (class=%d, confidence=%.1f%%)",
            label, class_id, confidence * 100,
        )

        # ── Step 2: Algorithm selection ────────────────────────────────────
        algo_key = self.CLASS_ALGORITHM_MAP[class_id]
        justification = self.JUSTIFICATIONS[class_id]

        # ── Step 3: Dynamic quantum tuning ────────────────────────────────
        if override_quantum is not None:
            quantum = override_quantum
        elif class_id == 1:       # I/O-bound: keep quantum small.
            quantum = max(2, self.base_quantum // 2)
        elif class_id == 2:       # Mixed: tune to burst variance.
            quantum = _compute_optimal_quantum(processes, self.base_quantum)
        else:
            quantum = self.base_quantum

        logger.info(
            "Selected algorithm: %s | Time quantum: %d",
            algo_key, quantum,
        )

        # ── Step 4: Execute scheduler ──────────────────────────────────────
        result = self._dispatch(algo_key, processes, quantum)
        result.update(self._compute_metrics(result["processes"]))

        # ── Step 5: Attach metadata ────────────────────────────────────────
        result["prediction"] = prediction
        result["selected_algorithm"] = result["algorithm"]
        result["justification"] = justification
        result["time_quantum"] = quantum
        result["workload_label"] = label
        result["confidence"] = confidence

        logger.info("Simulation complete. Context switches: %d", result["context_switches"])
        return result

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _dispatch(algo_key: str, processes: List[Process], quantum: int) -> Dict[str, Any]:
        """Run the specified algorithm and return raw results."""
        if algo_key == "SJF":
            return sjf_non_preemptive(processes)
        elif algo_key == "RR":
            return round_robin(processes, quantum)
        elif algo_key == "Priority":
            # If all processes have the same priority, fall back to Dynamic RR.
            unique_priorities = {p.priority for p in processes}
            if len(unique_priorities) <= 1:
                logger.info("Uniform priorities – falling back to Dynamic Round Robin.")
                return round_robin(processes, quantum)
            return priority_scheduling(processes)
        else:
            raise ValueError(f"Unknown algorithm key: {algo_key}")

    @staticmethod
    def _compute_metrics(procs: List[Process]) -> Dict[str, float]:
        """Compute and return aggregate metrics from a finished simulation."""
        n = len(procs)
        if n == 0:
            return {
                "avg_waiting_time": 0.0,
                "avg_turnaround_time": 0.0,
                "avg_response_time": 0.0,
                "throughput": 0.0,
            }

        completions = [p.completion_time for p in procs if p.completion_time is not None]
        arrivals = [p.arrival_time for p in procs]
        span = (max(completions) - min(arrivals)) if completions else 1

        return {
            "avg_waiting_time": round(sum(p.waiting_time for p in procs) / n, 3),
            "avg_turnaround_time": round(sum(p.turnaround_time for p in procs) / n, 3),
            "avg_response_time": round(sum(p.response_time for p in procs) / n, 3),
            "throughput": round(n / span if span > 0 else float(n), 4),
        }
