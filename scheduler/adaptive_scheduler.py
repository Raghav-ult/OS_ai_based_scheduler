"""
scheduler/adaptive_scheduler.py
---------------------------------
Adaptive AI-Driven CPU Scheduler implementations:

1. TrueAdaptiveScheduler — event-driven runtime adaptation on a Single Core:
   - Dynamic Ready-Queue Telemetry & Classification
   - Priority Aging & Anti-Starvation Safe Fallback Watchdog (Raghav's focus)
   - Re-evaluates policy at every arrival, completion, or quantum expiry
   - Returns timeline: List[(pid, start, end, policy_name)]

2. DualCoreAdaptiveScheduler — Dual-Core (SMP) Parallel Dispatcher:
   - Dispatches processes across Core 0 (Compute/Batch Core) and Core 1 (Interactive/Urgent Core)
   - Dynamic Work-Stealing: idle core assists busy core
   - Multi-core Starvation Watchdog
   - Returns timeline: List[(core_id, pid, start, end, policy_name)]

3. AdaptiveScheduler (legacy) — batch-level ML classifier for comparison.
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

POLICY_SRTF       = "SRTF"
POLICY_RR         = "Round Robin"
POLICY_PRIORITY   = "Priority"
POLICY_STARVATION = "Starvation Guard"

POLICY_COLORS: Dict[str, str] = {
    POLICY_SRTF:       "#667eea",   # indigo
    POLICY_RR:         "#f093fb",   # pink / magenta
    POLICY_PRIORITY:   "#fbbf24",   # amber
    POLICY_STARVATION: "#ef4444",   # red / alert (Anti-Starvation preemption)
    "Idle":            "#374151",   # slate
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
# 1. TrueAdaptiveScheduler (Single Core with Anti-Starvation Watchdog)
# ===========================================================================

class TrueAdaptiveScheduler:
    """
    Event-driven, runtime-adaptive CPU scheduler with Dynamic Priority Aging
    and an Anti-Starvation Deterministic Fallback Watchdog.

    At every scheduling decision point the algorithm:
        1. Checks for starving processes (continuous wait > starvation_threshold).
           If detected, deterministic fallback triggers: process is given immediate execution.
        2. Applies Dynamic Priority Aging to all waiting processes.
        3. Extracts ready-queue telemetry (variance, I/O ratio, priority spread).
        4. Dynamically switches between SRTF, Round Robin (dynamic Q), and Priority.
        5. Logs policy transitions and anti-starvation preemption events.
    """

    def __init__(
        self,
        base_quantum: int = 4,
        starvation_threshold: int = 15,
        enable_aging: bool = True,
        aging_interval: int = 4,
    ):
        self.base_quantum = base_quantum
        self.starvation_threshold = starvation_threshold
        self.enable_aging = enable_aging
        self.aging_interval = aging_interval

    @staticmethod
    def _queue_features(ready: List[Process]) -> Dict[str, float]:
        if not ready:
            return {}
        rem   = [p.remaining_time for p in ready]
        io    = [p.io_frequency   for p in ready]
        prios = [p.effective_priority for p in ready]
        n = len(rem)
        mean_rem = statistics.mean(rem)
        var_rem  = statistics.variance(rem) if n > 1 else 0.0
        return {
            "count":         n,
            "mean_rem":      mean_rem,
            "var_rem":       var_rem,
            "min_rem":       min(rem),
            "max_rem":       max(rem),
            "median_rem":    statistics.median(rem),
            "mean_io":       statistics.mean(io),
            "min_priority":  min(prios),
            "mean_priority": statistics.mean(prios),
        }

    def _select_policy(
        self,
        feat: Dict[str, float],
        ready: List[Process],
        current_time: int,
    ) -> Tuple[str, int, str, Optional[Process]]:
        """Returns (policy_name, quantum, human_readable_reason, target_process)."""
        if not feat or feat["count"] <= 1:
            return POLICY_SRTF, self.base_quantum, "Single process ready: SRTF execution optimal.", None

        # --- Rule 0: Anti-Starvation Safe Fallback Watchdog (Deterministic Guard) ---
        starving_procs = [
            p for p in ready
            if p.current_wait_since_active(current_time) >= self.starvation_threshold
        ]
        if starving_procs:
            starved = max(starving_procs, key=lambda p: p.current_wait_since_active(current_time))
            wait_ticks = starved.current_wait_since_active(current_time)
            reason = (
                f"⚠️ Anti-Starvation Watchdog Alert: P{starved.pid} waited {wait_ticks} ticks "
                f"without CPU access (threshold={self.starvation_threshold}). "
                f"Deterministic fallback triggered to guarantee fairness."
            )
            q = max(2, min(self.base_quantum, starved.remaining_time))
            return POLICY_STARVATION, q, reason, starved

        # Apply Aging to ready queue
        if self.enable_aging:
            for p in ready:
                p.update_effective_priority(current_time, self.aging_interval)

        var_rem     = feat["var_rem"]
        min_rem     = feat["min_rem"]
        median_rem  = feat["median_rem"]
        mean_rem    = feat["mean_rem"]
        mean_prio   = feat["mean_priority"]

        # Rule 1: Urgent Priority Task (Priority=1 or highly aged)
        urgent_proc = min(ready, key=lambda p: (p.effective_priority, p.remaining_time))
        if urgent_proc.effective_priority == 1 or (urgent_proc.effective_priority <= 2 and urgent_proc.effective_priority < mean_prio - 1.5):
            if urgent_proc.remaining_time <= min_rem or urgent_proc.effective_priority == 1:
                q = max(2, min(self.base_quantum, urgent_proc.remaining_time))
                reason = (f"Urgent task P{urgent_proc.pid} (effective priority={urgent_proc.effective_priority}, "
                          f"rem={urgent_proc.remaining_time}) detected; preempting with Priority policy.")
                return POLICY_PRIORITY, q, reason, urgent_proc

        # Rule 2: Interactive I/O Queue
        interactive_procs = [p for p in ready if p.io_frequency >= 0.60]
        if len(interactive_procs) >= 2 and min(p.remaining_time for p in interactive_procs) <= 4:
            q = max(2, min(self.base_quantum, int(median_rem)))
            reason = (f"Interactive workload ({len(interactive_procs)} I/O-intensive jobs ready); "
                      f"Round Robin with dynamic Q={q} guarantees low response latency.")
            return POLICY_RR, q, reason, None

        # Rule 3: Heterogeneous Bursts (Convoy Effect Prevention)
        if var_rem > 25.0:
            reason = (f"High burst variance ({var_rem:.1f}); SRTF preemptively clears short jobs "
                      f"to avoid convoy delays.")
        else:
            reason = (f"Compute-bound queue (avg remaining={mean_rem:.1f}, var={var_rem:.1f}); "
                      f"SRTF minimizes turnaround time.")
        return POLICY_SRTF, self.base_quantum, reason, None

    def run(
        self,
        processes: List[Process],
        base_quantum: Optional[int] = None,
    ) -> Dict[str, Any]:
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
        starvation_events = 0

        current_time   = 0
        completed_pids: set  = set()
        last_pid       = -1
        last_policy: Optional[str] = None

        rr_queue:    deque = deque()
        rr_enqueued: set   = set()
        rr_quantum         = self.base_quantum

        def ready_queue() -> List[Process]:
            return [p for p in procs
                    if p.pid not in completed_pids
                    and p.arrival_time <= current_time]

        def not_yet_arrived() -> List[Process]:
            return [p for p in procs
                    if p.pid not in completed_pids
                    and p.arrival_time > current_time]

        while len(completed_pids) < n:
            ready = ready_queue()

            if not ready:
                pending = not_yet_arrived()
                if not pending:
                    break
                current_time = min(p.arrival_time for p in pending)
                continue

            feat = self._queue_features(ready)
            policy, q, reason, target_proc = self._select_policy(feat, ready, current_time)

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
                rr_queue.clear()
                rr_enqueued.clear()

            # ── 0. Starvation Guard Preemption ───────────────────────────
            if policy == POLICY_STARVATION and target_proc is not None:
                chosen = target_proc
                chosen.starvation_rescues += 1
                starvation_events += 1

                if last_pid not in (-1, chosen.pid):
                    context_switches += 1
                if chosen.start_time is None:
                    chosen.start_time = current_time

                exec_time = min(q, chosen.remaining_time)
                start = current_time
                chosen.remaining_time -= exec_time
                current_time += exec_time
                last_pid = chosen.pid

                chosen.log_interval(start, current_time)
                timeline.append((chosen.pid, start, current_time, policy))

                if chosen.remaining_time == 0:
                    chosen.finalize(current_time)
                    completed_pids.add(chosen.pid)
                    last_pid = -1

            # ── 1. SRTF ──────────────────────────────────────────────────
            elif policy == POLICY_SRTF:
                chosen = min(ready, key=lambda p: (p.remaining_time, p.arrival_time, p.pid))

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

                chosen.log_interval(start, current_time)
                timeline.append((chosen.pid, start, current_time, policy))

                if chosen.remaining_time == 0:
                    chosen.finalize(current_time)
                    completed_pids.add(chosen.pid)
                    last_pid = -1

            # ── 2. Priority ──────────────────────────────────────────────
            elif policy == POLICY_PRIORITY:
                chosen = target_proc or min(ready, key=lambda p: (p.effective_priority, p.remaining_time, p.arrival_time, p.pid))

                if last_pid not in (-1, chosen.pid):
                    context_switches += 1
                if chosen.start_time is None:
                    chosen.start_time = current_time

                pending = not_yet_arrived()
                limit = q
                if pending:
                    next_ev = min(p.arrival_time for p in pending)
                    if next_ev > current_time:
                        limit = min(limit, next_ev - current_time)
                run_for = max(1, min(chosen.remaining_time, limit))

                start = current_time
                chosen.remaining_time -= run_for
                current_time += run_for
                last_pid = chosen.pid

                chosen.log_interval(start, current_time)
                timeline.append((chosen.pid, start, current_time, policy))

                if chosen.remaining_time == 0:
                    chosen.finalize(current_time)
                    completed_pids.add(chosen.pid)
                    last_pid = -1

            # ── 3. Round Robin ───────────────────────────────────────────
            elif policy == POLICY_RR:
                for p in sorted(ready, key=lambda x: (x.remaining_time, x.arrival_time)):
                    if p.pid not in rr_enqueued:
                        rr_queue.append(p)
                        rr_enqueued.add(p.pid)

                rr_queue = deque(p for p in rr_queue if p.pid not in completed_pids)
                if not rr_queue:
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

                chosen.log_interval(start, current_time)
                timeline.append((chosen.pid, start, current_time, policy))

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
                    rr_queue.append(chosen)

        collapsed = self._collapse(timeline)

        result = {
            "processes":          procs,
            "timeline":           collapsed,
            "adaptation_log":     adaptation_log,
            "context_switches":   context_switches,
            "starvation_rescues": starvation_events,
            "algorithm":          "True Adaptive AI Scheduler",
            "selected_algorithm": "True Adaptive AI Scheduler",
            "workload_label":     self._overall_label(processes),
            "confidence":         1.0,
            "architecture":       "Single Core",
            "justification": (
                "True Runtime Adaptation: Real-time telemetry inspects queue states, "
                "dynamically shifts between SRTF, RR, and Priority, while an active "
                "Starvation Watchdog guarantees zero starvation via deterministic fallback."
            ),
        }
        result.update(_aggregate_metrics(procs))
        return result

    @staticmethod
    def _collapse(
        timeline: List[Tuple[int, int, int, str]]
    ) -> List[Tuple[int, int, int, str]]:
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
# 2. DualCoreAdaptiveScheduler (Dual-Core SMP Parallel Dispatching)
# ===========================================================================

class DualCoreAdaptiveScheduler:
    """
    Symmetric Multiprocessing (SMP) Dual-Core Adaptive Scheduler.

    Core 0: Primary Compute Core (prioritizes compute-intensive, batch, and SRTF jobs).
    Core 1: Latency-Sensitive Core (prioritizes I/O-heavy, urgent priority, and RR).
    Work-Stealing: If either core is idle, it immediately assists the other queue.
    Starvation Watchdog runs globally across both cores.
    """

    def __init__(
        self,
        base_quantum: int = 4,
        starvation_threshold: int = 15,
        enable_aging: bool = True,
        aging_interval: int = 4,
    ):
        self.base_quantum = base_quantum
        self.starvation_threshold = starvation_threshold
        self.enable_aging = enable_aging
        self.aging_interval = aging_interval

    def run(
        self,
        processes: List[Process],
        base_quantum: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not processes:
            raise ValueError("Process list is empty.")
        if base_quantum is not None:
            self.base_quantum = base_quantum

        procs = _deep_reset(processes)
        n = len(procs)
        procs.sort(key=lambda p: (p.arrival_time, p.pid))

        timeline_core0: List[Tuple[int, int, int, str]] = []
        timeline_core1: List[Tuple[int, int, int, str]] = []
        combined_timeline: List[Tuple[int, int, int, int, str]] = []  # (core_id, pid, start, end, policy)
        adaptation_log: List[Dict] = []

        context_switches = 0
        starvation_rescues = 0

        # State tracking for Cores 0 and 1
        core_busy_until = [0, 0]
        core_proc: List[Optional[Process]] = [None, None]
        core_last_pid = [-1, -1]
        completed_pids: set = set()

        current_time = 0

        def ready_queue(t: int) -> List[Process]:
            running_pids = {p.pid for p in core_proc if p is not None}
            return [
                p for p in procs
                if p.pid not in completed_pids
                and p.pid not in running_pids
                and p.arrival_time <= t
                and p.remaining_time > 0
            ]

        while len(completed_pids) < n:
            # Advance time to earliest event (core release or next arrival)
            uncompleted = [p for p in procs if p.pid not in completed_pids]
            ready = ready_queue(current_time)

            # Check if any core finishes at current_time
            for c in range(2):
                if core_proc[c] is not None and core_busy_until[c] <= current_time:
                    p = core_proc[c]
                    if p.remaining_time <= 0:
                        p.finalize(current_time)
                        completed_pids.add(p.pid)
                    core_proc[c] = None

            ready = ready_queue(current_time)

            # If both cores are idle and ready queue is empty, advance clock
            if core_proc[0] is None and core_proc[1] is None and not ready:
                pending = [p for p in uncompleted if p.arrival_time > current_time]
                if not pending:
                    break
                current_time = min(p.arrival_time for p in pending)
                continue

            # Update Priority Aging on ready processes
            if self.enable_aging:
                for p in ready:
                    p.update_effective_priority(current_time, self.aging_interval)

            # Assign ready processes to idle cores
            for core_id in [0, 1]:
                if core_proc[core_id] is None:
                    ready = ready_queue(current_time)
                    if not ready:
                        continue

                    # 1. Starvation Watchdog Check (Top Priority)
                    starving = [
                        p for p in ready
                        if p.current_wait_since_active(current_time) >= self.starvation_threshold
                    ]
                    if starving:
                        chosen = max(starving, key=lambda p: p.current_wait_since_active(current_time))
                        policy = POLICY_STARVATION
                        q = max(2, min(self.base_quantum, chosen.remaining_time))
                        reason = (
                            f"Core {core_id} Anti-Starvation Watchdog: P{chosen.pid} starved for "
                            f"{chosen.current_wait_since_active(current_time)} ticks. Deterministic dispatch."
                        )
                        chosen.starvation_rescues += 1
                        starvation_rescues += 1

                    # 2. Core 0: Favours Compute / Batch (SRTF)
                    elif core_id == 0:
                        compute_candidates = [p for p in ready if p.io_frequency < 0.50]
                        if compute_candidates:
                            chosen = min(compute_candidates, key=lambda p: (p.remaining_time, p.arrival_time))
                            policy = POLICY_SRTF
                            q = chosen.remaining_time
                            reason = f"Core 0 (Compute Engine): Assigned compute task P{chosen.pid} via SRTF."
                        else:
                            # Work-stealing interactive task
                            chosen = ready[0]
                            policy = POLICY_RR
                            q = min(self.base_quantum, chosen.remaining_time)
                            reason = f"Core 0 (Work-Stealing): Assisted ready queue with P{chosen.pid}."

                    # 3. Core 1: Favours Interactive / Urgent Priority
                    else:
                        interactive_candidates = [
                            p for p in ready
                            if p.io_frequency >= 0.50 or p.effective_priority <= 2
                        ]
                        if interactive_candidates:
                            chosen = min(interactive_candidates, key=lambda p: (p.effective_priority, p.remaining_time))
                            policy = POLICY_PRIORITY if chosen.effective_priority <= 2 else POLICY_RR
                            q = min(self.base_quantum, chosen.remaining_time)
                            reason = f"Core 1 (Interactive Engine): Dispatched P{chosen.pid} under {policy}."
                        else:
                            # Work-stealing compute task
                            chosen = min(ready, key=lambda p: p.remaining_time)
                            policy = POLICY_SRTF
                            q = chosen.remaining_time
                            reason = f"Core 1 (Work-Stealing): Assisted Core 0 with compute task P{chosen.pid}."

                    if core_last_pid[core_id] != -1 and core_last_pid[core_id] != chosen.pid:
                        context_switches += 1
                    core_last_pid[core_id] = chosen.pid

                    if chosen.start_time is None:
                        chosen.start_time = current_time

                    exec_slice = min(q, chosen.remaining_time)
                    end_time = current_time + exec_slice

                    chosen.remaining_time -= exec_slice
                    chosen.log_interval(current_time, end_time)

                    core_proc[core_id] = chosen
                    core_busy_until[core_id] = end_time

                    timeline_entry = (chosen.pid, current_time, end_time, policy)
                    if core_id == 0:
                        timeline_core0.append(timeline_entry)
                    else:
                        timeline_core1.append(timeline_entry)

                    combined_timeline.append((core_id, chosen.pid, current_time, end_time, policy))

                    adaptation_log.append({
                        "time": current_time,
                        "core": f"Core {core_id}",
                        "policy": policy,
                        "event": f"Core {core_id} Dispatched P{chosen.pid}",
                        "reason": reason,
                        "queue_size": len(ready),
                        "queue_pids": [p.pid for p in ready],
                    })

            # Advance time to nearest future event
            future_events = [t for t in core_busy_until if t > current_time]
            pending_arrivals = [p.arrival_time for p in uncompleted if p.arrival_time > current_time]
            all_future = future_events + pending_arrivals

            if all_future:
                current_time = min(all_future)
            else:
                current_time += 1

        # Post-process metrics
        total_span = current_time
        core0_busy = sum(e[2] - e[1] for e in timeline_core0)
        core1_busy = sum(e[2] - e[1] for e in timeline_core1)

        util_core0 = round((core0_busy / total_span) * 100, 2) if total_span > 0 else 0.0
        util_core1 = round((core1_busy / total_span) * 100, 2) if total_span > 0 else 0.0
        overall_util = round(((core0_busy + core1_busy) / (2 * total_span)) * 100, 2) if total_span > 0 else 0.0

        result = {
            "processes": procs,
            "timeline": combined_timeline,
            "timeline_core0": timeline_core0,
            "timeline_core1": timeline_core1,
            "adaptation_log": adaptation_log,
            "context_switches": context_switches,
            "starvation_rescues": starvation_rescues,
            "algorithm": "Dual-Core Adaptive SMP",
            "selected_algorithm": "Dual-Core Adaptive SMP",
            "workload_label": TrueAdaptiveScheduler._overall_label(processes),
            "confidence": 1.0,
            "architecture": "Dual Core (SMP)",
            "makespan": total_span,
            "utilization_core0": util_core0,
            "utilization_core1": util_core1,
            "overall_utilization": overall_util,
            "justification": (
                "Dual-Core SMP Dispatching: Dynamically load balances compute and interactive queues "
                "across Core 0 and Core 1 with automated work-stealing and an active anti-starvation watchdog."
            ),
        }
        result.update(_aggregate_metrics(procs))
        return result


# ===========================================================================
# 3. AdaptiveScheduler (legacy — batch-level ML, single static algorithm)
# ===========================================================================

class AdaptiveScheduler:
    """
    Batch-level ML classifier: predicts workload class at t=0 and runs
    one fixed algorithm for the entire simulation. Kept for comparison.
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
