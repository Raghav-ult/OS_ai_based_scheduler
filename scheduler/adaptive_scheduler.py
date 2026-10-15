"""
scheduler/adaptive_scheduler.py
---------------------------------
Two scheduler implementations:

1. TrueAdaptiveScheduler (primary) — event-driven runtime adaptation:
   - Re-evaluates the scheduling policy at every significant event
     (process arrival, process completion, quantum expiry).
   - Extracts live ready-queue features and applies a decision tree to
     pick the best algorithm for the *current* workload snapshot.
   - Logs every policy transition for dashboard visualization.
   - Returns 4-tuple timeline: (pid, start, end, policy_name).

2. AdaptiveScheduler (legacy) — batch-level ML classifier (kept for
   reference in single-batch comparison runs).
"""

import copy
import math
import statistics
import logging
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

from core.process import Process
from core.schedulers import sjf_non_preemptive, round_robin, priority_scheduling
from ml.predictor import predict_workload

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Scheduler] %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Shared policy constants (imported by dashboard.py for colouring)
# ---------------------------------------------------------------------------

POLICY_SRTF     = "SRTF"
POLICY_RR       = "Round Robin"
POLICY_PRIORITY = "Priority"

POLICY_COLORS: Dict[str, str] = {
    POLICY_SRTF:     "#667eea",   # indigo
    POLICY_RR:       "#f093fb",   # pink / magenta
    POLICY_PRIORITY: "#fbbf24",   # amber
    "Idle":          "#374151",   # slate
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_reset(processes: List[Process]) -> List[Process]:
    """Return deep-copied, reset process objects."""
    cloned = [copy.deepcopy(p) for p in processes]
    for p in cloned:
        p.reset()
    return cloned


def _aggregate_metrics(procs: List[Process]) -> Dict[str, float]:
    n = len(procs)
    if n == 0:
        return dict(avg_waiting_time=0.0, avg_turnaround_time=0.0,
                    avg_response_time=0.0, throughput=0.0)
    completions = [p.completion_time for p in procs if p.completion_time is not None]
    arrivals    = [p.arrival_time    for p in procs]
    span = (max(completions) - min(arrivals)) if completions else 1
    return {
        "avg_waiting_time":    round(sum(p.waiting_time    for p in procs) / n, 3),
        "avg_turnaround_time": round(sum(p.turnaround_time for p in procs) / n, 3),
        "avg_response_time":   round(sum(p.response_time   for p in procs) / n, 3),
        "throughput":          round(n / span if span > 0 else float(n), 4),
    }


# ===========================================================================
# 1. TrueAdaptiveScheduler
# ===========================================================================

class TrueAdaptiveScheduler:
    """
    Event-driven, runtime-adaptive CPU scheduler.

    At every scheduling decision point the algorithm:
        1. Inspects the current ready queue.
        2. Extracts real-time features (remaining bursts, I/O ratio, priority).
        3. Applies a decision-tree heuristic to select the optimal policy.
        4. Runs the chosen process for one bounded interval.
        5. Logs any policy change as an adaptation event.

    Policy decision hierarchy
    ─────────────────────────
    Rule 1  urgent high-priority task in queue (priority <= 2)
            → Priority Scheduling
    Rule 2  I/O-dominated queue (mean io_freq > 0.45)
            → Round Robin  with dynamic Q = max(2, median(remaining))
    Rule 3  short interactive jobs (mean_rem < 8, ≥ 2 processes)
            → Round Robin  with dynamic Q = max(2, median(remaining))
    Rule 4  high burst variance (var > 50)
            → SRTF  (minimises turnaround across heterogeneous bursts)
    Rule 5  default / CPU-bound
            → SRTF

    Timeline format: List[(pid, start, end, policy_name)]
    """

    def __init__(self, base_quantum: int = 4):
        self.base_quantum = base_quantum

    # ── Feature extraction ──────────────────────────────────────────────

    @staticmethod
    def _queue_features(ready: List[Process]) -> Dict[str, float]:
        if not ready:
            return {}
        rem   = [p.remaining_time for p in ready]
        io    = [p.io_frequency   for p in ready]
        prios = [p.priority        for p in ready]
        n = len(rem)
        return {
            "count":        n,
            "mean_rem":     statistics.mean(rem),
            "var_rem":      statistics.variance(rem) if n > 1 else 0.0,
            "median_rem":   statistics.median(rem),
            "mean_io":      statistics.mean(io),
            "min_priority": min(prios),
        }

    # ── Policy decision tree ─────────────────────────────────────────────

    def _select_policy(self, feat: Dict[str, float]) -> Tuple[str, int, str]:
        """Returns (policy_name, quantum, human_readable_reason)."""
        if not feat:
            return POLICY_SRTF, self.base_quantum, "Default (no features available)"

        mean_rem   = feat["mean_rem"]
        var_rem    = feat["var_rem"]
        median_rem = feat["median_rem"]
        mean_io    = feat["mean_io"]
        min_prio   = feat["min_priority"]
        count      = feat["count"]

        if min_prio <= 2 and count > 1:
            reason = (f"Urgent high-priority process (priority={int(min_prio)}) "
                      f"detected; preempting with Priority Scheduling.")
            return POLICY_PRIORITY, self.base_quantum, reason

        if mean_io > 0.45:
            q = max(2, int(median_rem))
            reason = (f"I/O-dominated queue (avg I/O freq={mean_io:.2f}); "
                      f"Round Robin with dynamic Q={q} for low response time.")
            return POLICY_RR, q, reason

        if mean_rem < 8 and count >= 2:
            q = max(2, int(median_rem))
            reason = (f"Short interactive jobs (avg remaining={mean_rem:.1f}); "
                      f"Round Robin with dynamic Q={q}.")
            return POLICY_RR, q, reason

        if var_rem > 50:
            reason = (f"High burst variance ({var_rem:.1f}); "
                      f"SRTF minimises average turnaround time.")
            return POLICY_SRTF, self.base_quantum, reason

        reason = (f"CPU-bound queue (avg remaining={mean_rem:.1f}, "
                  f"var={var_rem:.1f}); SRTF is optimal.")
        return POLICY_SRTF, self.base_quantum, reason

    # ── Main simulation ──────────────────────────────────────────────────

    def run(
        self,
        processes: List[Process],
        base_quantum: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute true runtime-adaptive scheduling and return full results.

        Returns
        -------
        dict containing:
            processes         – List[Process] with filled metrics
            timeline          – List[(pid, start, end, policy)] 4-tuples
            adaptation_log    – List[dict] of policy-transition events
            context_switches  – int
            avg_waiting_time  – float
            avg_turnaround_time – float
            avg_response_time – float
            throughput        – float
            algorithm         – "True Adaptive AI Scheduler"
        """
        if not processes:
            raise ValueError("Process list is empty.")
        if base_quantum is not None:
            self.base_quantum = base_quantum

        procs = _deep_reset(processes)
        n = len(procs)
        procs.sort(key=lambda p: (p.arrival_time, p.pid))

        timeline:       List[Tuple[int, int, int, str]] = []
        adaptation_log: List[Dict]                       = []
        context_switches = 0

        current_time   = 0
        completed_pids: set  = set()
        last_pid       = -1
        last_policy: Optional[str] = None

        # RR state (rebuilt on every policy switch to RR)
        rr_queue:    deque = deque()
        rr_enqueued: set   = set()
        rr_quantum         = self.base_quantum

        # ── Closures ────────────────────────────────────────────────────
        def ready_queue() -> List[Process]:
            return [p for p in procs
                    if p.pid not in completed_pids
                    and p.arrival_time <= current_time]

        def not_yet_arrived() -> List[Process]:
            return [p for p in procs
                    if p.pid not in completed_pids
                    and p.arrival_time > current_time]

        # ── Simulation loop ──────────────────────────────────────────────
        while len(completed_pids) < n:
            ready = ready_queue()

            # Idle: advance clock to next arrival
            if not ready:
                pending = not_yet_arrived()
                if not pending:
                    break
                current_time = min(p.arrival_time for p in pending)
                continue

            # ── Policy (re-)evaluation ───────────────────────────────────
            feat = self._queue_features(ready)
            policy, q, reason = self._select_policy(feat)

            if policy != last_policy:
                adaptation_log.append({
                    "time":       current_time,
                    "policy":     policy,
                    "event":      f"Switch to {policy}",
                    "reason":     reason,
                    "queue_size": len(ready),
                    "queue_pids": [p.pid for p in ready],
                })
                last_policy  = policy
                rr_quantum   = q
                # Rebuild RR state fresh on every policy switch
                rr_queue.clear()
                rr_enqueued.clear()

            # ── SRTF  (event-driven, not tick-by-tick) ───────────────────
            if policy == POLICY_SRTF:
                chosen = min(ready,
                             key=lambda p: (p.remaining_time, p.arrival_time, p.pid))

                if last_pid not in (-1, chosen.pid):
                    context_switches += 1
                if chosen.start_time is None:
                    chosen.start_time = current_time

                pending = not_yet_arrived()
                if pending:
                    next_arr = min(p.arrival_time for p in pending)
                    run_for  = min(chosen.remaining_time, next_arr - current_time)
                else:
                    run_for  = chosen.remaining_time
                run_for = max(1, run_for)

                start = current_time
                chosen.remaining_time -= run_for
                current_time += run_for
                last_pid = chosen.pid

                timeline.append((chosen.pid, start, current_time, policy))

                if chosen.remaining_time == 0:
                    chosen.finalize(current_time)
                    completed_pids.add(chosen.pid)
                    last_pid = -1

            # ── Priority  (preempts if higher-priority process arrives) ───
            elif policy == POLICY_PRIORITY:
                chosen = min(ready,
                             key=lambda p: (p.priority, p.arrival_time, p.pid))

                if last_pid not in (-1, chosen.pid):
                    context_switches += 1
                if chosen.start_time is None:
                    chosen.start_time = current_time

                pending = not_yet_arrived()
                if pending:
                    urgent = [p for p in pending if p.priority < chosen.priority]
                    if urgent:
                        next_ev = min(p.arrival_time for p in urgent)
                    else:
                        next_ev = min(p.arrival_time for p in pending)
                    run_for = min(chosen.remaining_time, next_ev - current_time)
                else:
                    run_for = chosen.remaining_time
                run_for = max(1, run_for)

                start = current_time
                chosen.remaining_time -= run_for
                current_time += run_for
                last_pid = chosen.pid

                timeline.append((chosen.pid, start, current_time, policy))

                if chosen.remaining_time == 0:
                    chosen.finalize(current_time)
                    completed_pids.add(chosen.pid)
                    last_pid = -1

            # ── Round Robin ───────────────────────────────────────────────
            elif policy == POLICY_RR:
                # Admit all currently-ready processes not yet in the queue
                for p in ready:
                    if p.pid not in rr_enqueued:
                        rr_queue.append(p)
                        rr_enqueued.add(p.pid)

                # Purge completed
                rr_queue = deque(p for p in rr_queue
                                 if p.pid not in completed_pids)

                if not rr_queue:          # safety re-seed
                    for p in ready:
                        rr_queue.append(p)
                        rr_enqueued.add(p.pid)

                chosen = rr_queue.popleft()

                if last_pid not in (-1, chosen.pid):
                    context_switches += 1
                if chosen.start_time is None:
                    chosen.start_time = current_time

                start     = current_time
                exec_time = min(rr_quantum, chosen.remaining_time)
                chosen.remaining_time -= exec_time
                current_time += exec_time
                last_pid = chosen.pid

                timeline.append((chosen.pid, start, current_time, policy))

                # Admit processes that arrived during this quantum
                for p in procs:
                    if (p.pid not in completed_pids
                            and p.pid not in rr_enqueued
                            and p.arrival_time <= current_time):
                        rr_queue.append(p)
                        rr_enqueued.add(p.pid)

                if chosen.remaining_time == 0:
                    chosen.finalize(current_time)
                    completed_pids.add(chosen.pid)
                    last_pid = -1
                else:
                    rr_queue.append(chosen)     # re-enqueue at back

        # ── Post-process ─────────────────────────────────────────────────
        collapsed = self._collapse(timeline)

        for pid, start, end, pol in collapsed:
            proc = next((p for p in procs if p.pid == pid), None)
            if proc:
                proc.log_interval(start, end)

        result = {
            "processes":          procs,
            "timeline":           collapsed,
            "adaptation_log":     adaptation_log,
            "context_switches":   context_switches,
            "algorithm":          "True Adaptive AI Scheduler",
            "selected_algorithm": "True Adaptive AI Scheduler",
            "workload_label":     self._overall_label(processes),
            "confidence":         1.0,
            "justification": (
                "True Runtime Adaptation: the scheduling policy is re-evaluated "
                "at every process arrival and completion event, based on live "
                "ready-queue characteristics (remaining bursts, I/O frequency, priority)."
            ),
        }
        result.update(_aggregate_metrics(procs))
        return result

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _collapse(
        timeline: List[Tuple[int, int, int, str]]
    ) -> List[Tuple[int, int, int, str]]:
        """Merge adjacent same-(pid, policy) intervals for a cleaner Gantt."""
        if not timeline:
            return []
        result = []
        cur_pid, cur_s, cur_e, cur_pol = timeline[0]
        for pid, s, e, pol in timeline[1:]:
            if pid == cur_pid and pol == cur_pol and s == cur_e:
                cur_e = e
            else:
                result.append((cur_pid, cur_s, cur_e, cur_pol))
                cur_pid, cur_s, cur_e, cur_pol = pid, s, e, pol
        result.append((cur_pid, cur_s, cur_e, cur_pol))
        return result

    @staticmethod
    def _overall_label(processes: List[Process]) -> str:
        try:
            return predict_workload(processes)["label"]
        except Exception:
            return "Mixed"


# ===========================================================================
# 2.  AdaptiveScheduler  (legacy — batch-level ML, single static algorithm)
# ===========================================================================

class AdaptiveScheduler:
    """
    Batch-level ML classifier: predicts workload class at t=0 and runs
    one fixed algorithm for the entire simulation.  Kept for comparison.
    """

    CLASS_ALGORITHM_MAP = {0: "SJF", 1: "RR", 2: "Priority"}
    JUSTIFICATIONS = {
        0: "CPU-bound: SJF minimises average turnaround time.",
        1: "I/O-bound: Round Robin ensures low response time.",
        2: "Mixed: Priority Scheduling serves urgent processes first.",
    }

    def __init__(self, base_quantum: int = 4):
        self.base_quantum = base_quantum

    def run(
        self,
        processes: List[Process],
        override_quantum: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not processes:
            raise ValueError("Cannot schedule an empty process list.")

        prediction    = predict_workload(processes)
        class_id      = prediction["class_id"]
        algo_key      = self.CLASS_ALGORITHM_MAP[class_id]
        quantum       = override_quantum or self.base_quantum
        if class_id == 1:
            quantum = max(2, quantum // 2)

        result = self._dispatch(algo_key, processes, quantum)
        result.update(_aggregate_metrics(result["processes"]))
        result["prediction"]         = prediction
        result["selected_algorithm"] = result["algorithm"]
        result["justification"]      = self.JUSTIFICATIONS[class_id]
        result["time_quantum"]       = quantum
        result["workload_label"]     = prediction["label"]
        result["confidence"]         = prediction["confidence"]
        return result

    @staticmethod
    def _dispatch(algo_key, processes, quantum):
        if algo_key == "SJF":
            return sjf_non_preemptive(processes)
        if algo_key == "RR":
            return round_robin(processes, quantum)
        if len({p.priority for p in processes}) <= 1:
            return round_robin(processes, quantum)
        return priority_scheduling(processes)
