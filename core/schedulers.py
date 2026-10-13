"""
core/schedulers.py
------------------
Full implementations of four classic CPU scheduling algorithms:
    1. First-Come First-Served (FCFS)
    2. Shortest Job First — Non-preemptive and Preemptive (SRTF)
    3. Round Robin (RR) with configurable time quantum
    4. Priority Scheduling (Non-preemptive, lower number = higher priority)

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

    Returns:
        dict with keys: 'processes', 'timeline', 'context_switches'
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

    Returns:
        dict with keys: 'processes', 'timeline', 'context_switches'
    """
    procs = _deep_copy_processes(processes)

    current_time = 0
    context_switches = 0
    timeline: Timeline = []
    last_pid = -1
    completed: List[Process] = []
    remaining = procs[:]

    while remaining:
        # Gather processes that have arrived by current_time.
        available = [p for p in remaining if p.arrival_time <= current_time]

        if not available:
            # Advance clock to the earliest arriving process.
            current_time = min(p.arrival_time for p in remaining)
            continue

        # Pick the shortest burst time; break ties by arrival time then pid.
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

    At every time tick the ready queue is rescanned; if a newly arrived
    process has a shorter remaining time than the current one, a preemption
    occurs.

    Returns:
        dict with keys: 'processes', 'timeline', 'context_switches'
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
        # All processes that have arrived.
        available = [p for p in procs if p.arrival_time <= current_time and p.remaining_time > 0]

        if not available:
            current_time += 1
            continue

        # Pick shortest remaining time.
        chosen = min(available, key=lambda p: (p.remaining_time, p.arrival_time, p.pid))

        if chosen.pid != last_pid:
            # Flush previous interval.
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
            # Flush the last interval and finalize.
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

    Processes cycle through the ready queue in order of arrival;
    each gets at most `time_quantum` ticks per turn.

    Args:
        time_quantum: Maximum CPU slice per scheduling turn (ticks).

    Returns:
        dict with keys: 'processes', 'timeline', 'context_switches'
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

    # Index to track which processes have been enqueued.
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

    # Seed with processes arriving at t=0.
    enqueue_arrivals(current_time)

    # If nothing arrived at 0, jump to first arrival.
    if not ready_queue and enqueue_idx < n:
        current_time = all_procs_sorted[enqueue_idx].arrival_time
        enqueue_arrivals(current_time)

    while completed_count < n:
        if not ready_queue:
            # Idle: advance to next arrival.
            if enqueue_idx < n:
                current_time = all_procs_sorted[enqueue_idx].arrival_time
                enqueue_arrivals(current_time)
            continue

        proc = ready_queue.pop(0)

        if last_pid != -1 and last_pid != proc.pid:
            context_switches += 1
        last_pid = proc.pid

        # Determine actual execution slice.
        exec_time = min(time_quantum, proc.remaining_time)
        start = current_time
        end = current_time + exec_time

        proc.log_interval(start, end)
        timeline.append((proc.pid, start, end))

        proc.remaining_time -= exec_time
        current_time = end

        # Enqueue any processes that arrived during this slice.
        enqueue_arrivals(current_time)

        if proc.remaining_time == 0:
            proc.finalize(current_time)
            completed_count += 1
        else:
            # Re-enqueue at the back of the queue.
            ready_queue.append(proc)

    return {
        "processes": procs,
        "timeline": timeline,
        "context_switches": context_switches,
        "algorithm": f"Round Robin (Q={time_quantum})",
    }


# ---------------------------------------------------------------------------
# 5. Priority Scheduling (Non-preemptive, lower number = higher priority)
# ---------------------------------------------------------------------------

def priority_scheduling(processes: List[Process]) -> Dict[str, Any]:
    """
    Non-preemptive Priority Scheduler.

    At each scheduling decision, selects the highest-priority process
    (lowest `priority` value) from those that have arrived.
    Ties broken by arrival time then pid.

    Returns:
        dict with keys: 'processes', 'timeline', 'context_switches'
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

        # Lower priority number = higher urgency.
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

    return {
        "processes": completed,
        "timeline": timeline,
        "context_switches": context_switches,
        "algorithm": "Priority Scheduling",
    }
