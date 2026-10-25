"""
core/schedulers.py
------------------
Full implementations of classic CPU scheduling algorithms:
    1. First-Come First-Served (FCFS)
    2. Shortest Job First — Non-preemptive and Preemptive (SRTF)
    3. Round Robin (RR) with configurable time quantum
    4. Priority Scheduling (Non-preemptive, lower number = higher priority)
    5. Priority Scheduling with Dynamic Aging (Anti-Starvation)

Each function accepts a list of Process objects, runs the simulation
from scratch (calling process.reset()), and returns:
    - updated Process objects (with metrics filled in)
    - execution_timeline: list of (pid, start, end) tuples
    - context_switches: total number of context switches that occurred
"""

import copy
from typing import List, Tuple, Dict, Any

from core.process import Process


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Timeline = List[Tuple[int, int, int]]   # (pid, start, end)


def _deep_copy_processes(processes: List[Process]) -> List[Process]:
    """Return deep-copied, reset Process objects so originals are never mutated."""
    cloned = [copy.deepcopy(p) for p in processes]
    for p in cloned:
        p.reset()
    return cloned


# ---------------------------------------------------------------------------
# 1. First-Come First-Served (FCFS)
# ---------------------------------------------------------------------------

def fcfs(processes: List[Process]) -> Dict[str, Any]:
    """
    Non-preemptive First-Come First-Served scheduler.

    Processes are sorted by arrival time and executed to completion
    before the next one starts.
    """
    procs = _deep_copy_processes(processes)
    procs.sort(key=lambda p: (p.arrival_time, p.pid))

    current_time = 0
    context_switches = 0
    timeline: Timeline = []
    last_pid = -1

    for proc in procs:
        # CPU idles if the next process hasn't arrived yet.
        if current_time < proc.arrival_time:
            current_time = proc.arrival_time

        start = current_time
        end = current_time + proc.burst_time

        proc.log_interval(start, end)
        proc.finalize(end)

        timeline.append((proc.pid, start, end))

        if last_pid != -1 and last_pid != proc.pid:
            context_switches += 1
        last_pid = proc.pid

        current_time = end

    return {
        "processes": procs,
        "timeline": timeline,
        "context_switches": context_switches,
        "algorithm": "FCFS",
    }


# ---------------------------------------------------------------------------
# 2. Shortest Job First — Non-preemptive
# ---------------------------------------------------------------------------

def sjf_non_preemptive(processes: List[Process]) -> Dict[str, Any]:
    """
    Non-preemptive Shortest Job First scheduler.

    At each scheduling decision point, the process with the shortest
    burst_time among all arrived processes is selected.
    """
    procs = _deep_copy_processes(processes)

    current_time = 0
    context_switches = 0
    timeline: Timeline = []
    last_pid = -1
    completed: List[Process] = []
    remaining = procs[:]

    while remaining:
        available = [p for p in remaining if p.arrival_time <= current_time]

        if not available:
            current_time = min(p.arrival_time for p in remaining)
            continue

        chosen = min(available, key=lambda p: (p.burst_time, p.arrival_time, p.pid))
        remaining.remove(chosen)

        start = current_time
        end = current_time + chosen.burst_time

        chosen.log_interval(start, end)
        chosen.finalize(end)

        timeline.append((chosen.pid, start, end))

        if last_pid != -1 and last_pid != chosen.pid:
            context_switches += 1
        last_pid = chosen.pid

        current_time = end
        completed.append(chosen)

    return {
        "processes": completed,
        "timeline": timeline,
        "context_switches": context_switches,
        "algorithm": "SJF (Non-Preemptive)",
    }


# ---------------------------------------------------------------------------
# 3. Shortest Remaining Time First (Preemptive SJF)
# ---------------------------------------------------------------------------

def sjf_preemptive(processes: List[Process]) -> Dict[str, Any]:
    """
    Preemptive Shortest Job First (Shortest Remaining Time First – SRTF).
    """
    procs = _deep_copy_processes(processes)
    procs.sort(key=lambda p: p.arrival_time)

    n = len(procs)
    current_time = 0
    context_switches = 0
    timeline: Timeline = []
    completed_count = 0
    last_pid = -1
    current_proc: Process | None = None
    interval_start = 0

    while completed_count < n:
        available = [p for p in procs if p.arrival_time <= current_time and p.remaining_time > 0]

        if not available:
            current_time += 1
            continue

        chosen = min(available, key=lambda p: (p.remaining_time, p.arrival_time, p.pid))

        if chosen.pid != last_pid:
            if current_proc is not None and interval_start < current_time:
                current_proc.log_interval(interval_start, current_time)
                timeline.append((current_proc.pid, interval_start, current_time))
            if last_pid != -1:
                context_switches += 1
            interval_start = current_time
            last_pid = chosen.pid
            current_proc = chosen

        chosen.remaining_time -= 1
        current_time += 1

        if chosen.remaining_time == 0:
            chosen.log_interval(interval_start, current_time)
            timeline.append((chosen.pid, interval_start, current_time))
            chosen.finalize(current_time)
            completed_count += 1
            last_pid = -1
            current_proc = None

    return {
        "processes": procs,
        "timeline": timeline,
        "context_switches": context_switches,
        "algorithm": "SJF (Preemptive / SRTF)",
    }


# ---------------------------------------------------------------------------
# 4. Round Robin
# ---------------------------------------------------------------------------

def round_robin(processes: List[Process], time_quantum: int = 4) -> Dict[str, Any]:
    """
    Preemptive Round Robin scheduler with a configurable time quantum.
    """
    if time_quantum <= 0:
        raise ValueError("time_quantum must be a positive integer.")

    procs = _deep_copy_processes(processes)
    procs.sort(key=lambda p: (p.arrival_time, p.pid))

    current_time = 0
    context_switches = 0
    timeline: Timeline = []
    ready_queue: List[Process] = []
    last_pid = -1

    all_procs_sorted = sorted(procs, key=lambda p: (p.arrival_time, p.pid))
    enqueue_idx = 0
    n = len(procs)
    completed_count = 0

    def enqueue_arrivals(t: int):
        nonlocal enqueue_idx
        while enqueue_idx < n and all_procs_sorted[enqueue_idx].arrival_time <= t:
            if all_procs_sorted[enqueue_idx].remaining_time > 0:
                ready_queue.append(all_procs_sorted[enqueue_idx])
            enqueue_idx += 1

    enqueue_arrivals(current_time)

    if not ready_queue and enqueue_idx < n:
        current_time = all_procs_sorted[enqueue_idx].arrival_time
        enqueue_arrivals(current_time)

    while completed_count < n:
        if not ready_queue:
            if enqueue_idx < n:
                current_time = all_procs_sorted[enqueue_idx].arrival_time
                enqueue_arrivals(current_time)
            continue

        proc = ready_queue.pop(0)

        if last_pid != -1 and last_pid != proc.pid:
            context_switches += 1
        last_pid = proc.pid

        exec_time = min(time_quantum, proc.remaining_time)
        start = current_time
        end = current_time + exec_time

        proc.log_interval(start, end)
        timeline.append((proc.pid, start, end))

        proc.remaining_time -= exec_time
        current_time = end

        enqueue_arrivals(current_time)

        if proc.remaining_time == 0:
            proc.finalize(current_time)
            completed_count += 1
        else:
            ready_queue.append(proc)

    return {
        "processes": procs,
        "timeline": timeline,
        "context_switches": context_switches,
        "algorithm": f"Round Robin (Q={time_quantum})",
    }


# ---------------------------------------------------------------------------
# 5. Priority Scheduling (Static vs Dynamic Aging)
# ---------------------------------------------------------------------------

def priority_scheduling(
    processes: List[Process],
    enable_aging: bool = False,
    aging_interval: int = 4,
) -> Dict[str, Any]:
    """
    Priority Scheduler.
    If enable_aging is True, waiting processes gain 1 priority level every `aging_interval` ticks,
    preventing lower-priority processes from starving indefinitely.
    """
    procs = _deep_copy_processes(processes)

    current_time = 0
    context_switches = 0
    timeline: Timeline = []
    last_pid = -1
    completed: List[Process] = []
    remaining = procs[:]

    while remaining:
        available = [p for p in remaining if p.arrival_time <= current_time]

        if not available:
            current_time = min(p.arrival_time for p in remaining)
            continue

        if enable_aging:
            for p in available:
                p.update_effective_priority(current_time, aging_interval)
            chosen = min(available, key=lambda p: (p.effective_priority, p.arrival_time, p.pid))
        else:
            chosen = min(available, key=lambda p: (p.priority, p.arrival_time, p.pid))

        remaining.remove(chosen)

        start = current_time
        end = current_time + chosen.burst_time

        chosen.log_interval(start, end)
        chosen.finalize(end)

        timeline.append((chosen.pid, start, end))

        if last_pid != -1 and last_pid != chosen.pid:
            context_switches += 1
        last_pid = chosen.pid

        current_time = end
        completed.append(chosen)

    algo_name = f"Priority (Aging Every {aging_interval}t)" if enable_aging else "Priority (Static)"
    return {
        "processes": completed,
        "timeline": timeline,
        "context_switches": context_switches,
        "algorithm": algo_name,
    }


def priority_aging(processes: List[Process], aging_interval: int = 4) -> Dict[str, Any]:
    """Explicit helper to run Priority Scheduling with Anti-Starvation Aging enabled."""
    return priority_scheduling(processes, enable_aging=True, aging_interval=aging_interval)
