"""
core/process.py
---------------
Defines the Process class representing a single schedulable unit.
Tracks all scheduling metrics, dynamic priority aging, starvation metrics,
and execution intervals for Gantt chart rendering.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Process:
    """
    Represents a single process in the CPU scheduling simulator.

    Attributes:
        pid            : Unique process identifier.
        arrival_time   : Time at which the process arrives in the ready queue.
        burst_time     : Total CPU time required by the process.
        priority       : Priority value (lower number = higher priority).
        io_frequency   : Estimated fraction of time spent doing I/O (0.0 – 1.0).

    Computed at runtime:
        remaining_time : CPU burst time still left (updated during simulation).
        start_time     : Wall-clock tick when the process first touches the CPU.
        completion_time: Wall-clock tick when the process finishes execution.
        waiting_time   : Total time spent in the ready queue.
        turnaround_time: completion_time - arrival_time.
        response_time  : start_time - arrival_time.
        last_cpu_time  : Wall-clock tick when process last relinquished or touched CPU.
        effective_priority: Dynamically adjusted priority taking aging into account.
        peak_effective_priority: Best (lowest number) priority reached via aging.
        starvation_rescues: Number of times this process was boosted by anti-starvation guard.
    """

    # --- Identity & input parameters ---
    pid: int
    arrival_time: int
    burst_time: int
    priority: int = 0
    io_frequency: float = 0.0          # 0.0 = pure CPU-bound, 1.0 = heavily I/O-bound

    # --- Runtime state ---
    remaining_time: int = field(init=False)
    start_time: Optional[int] = field(default=None, init=False)
    completion_time: Optional[int] = field(default=None, init=False)
    last_cpu_time: int = field(default=0, init=False)
    effective_priority: int = field(default=0, init=False)
    peak_effective_priority: int = field(default=0, init=False)
    starvation_rescues: int = field(default=0, init=False)

    # --- Derived scheduling metrics (populated after simulation) ---
    waiting_time: float = field(default=0.0, init=False)
    turnaround_time: float = field(default=0.0, init=False)
    response_time: float = field(default=0.0, init=False)

    # --- Execution history for Gantt chart rendering ---
    # Each tuple is (start_tick, end_tick) on the CPU.
    execution_intervals: List[Tuple[int, int]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.remaining_time = self.burst_time
        self.effective_priority = self.priority
        self.peak_effective_priority = self.priority
        self.last_cpu_time = self.arrival_time

    # ------------------------------------------------------------------
    # Priority Aging & Anti-Starvation Helpers
    # ------------------------------------------------------------------

    def update_effective_priority(self, current_time: int, aging_interval: int = 4) -> int:
        """
        Dynamically age the process based on time spent waiting in the ready queue.
        Lower number = higher priority. Priority improves by 1 every aging_interval ticks.
        Records peak_effective_priority so historical aged priority is preserved.
        """
        if current_time <= self.arrival_time:
            self.effective_priority = self.priority
            self.peak_effective_priority = min(self.peak_effective_priority, self.effective_priority)
            return self.effective_priority

        wait_duration = current_time - max(self.arrival_time, self.last_cpu_time)
        priority_boost = int(wait_duration // max(1, aging_interval))
        self.effective_priority = max(1, self.priority - priority_boost)
        self.peak_effective_priority = min(self.peak_effective_priority, self.effective_priority)
        return self.effective_priority

    def current_wait_since_active(self, current_time: int) -> int:
        """Returns the continuous duration this process has waited without touching CPU."""
        if current_time <= self.arrival_time:
            return 0
        return current_time - max(self.arrival_time, self.last_cpu_time)

    # ------------------------------------------------------------------
    # Execution history helpers
    # ------------------------------------------------------------------

    def log_interval(self, start: int, end: int) -> None:
        """Record a CPU execution interval (start, end) for Gantt rendering."""
        if start >= end:
            return
        self.execution_intervals.append((start, end))
        self.last_cpu_time = end

        # Track first time the process touched the CPU.
        if self.start_time is None:
            self.start_time = start

    def finalize(self, current_time: int) -> None:
        """
        Call once the process has finished execution.
        Computes and stores all derived scheduling metrics.
        """
        self.completion_time = current_time
        self.turnaround_time = self.completion_time - self.arrival_time
        self.waiting_time = self.turnaround_time - self.burst_time
        if self.start_time is not None:
            self.response_time = self.start_time - self.arrival_time
        else:
            self.response_time = 0.0

        # Clamp to zero to handle floating-point edge-cases.
        self.waiting_time = max(0.0, self.waiting_time)
        self.response_time = max(0.0, self.response_time)

    def reset(self) -> None:
        """Reset all runtime state so the same process object can be re-simulated."""
        self.remaining_time = self.burst_time
        self.start_time = None
        self.completion_time = None
        self.last_cpu_time = self.arrival_time
        self.effective_priority = self.priority
        self.peak_effective_priority = self.priority
        self.starvation_rescues = 0
        self.waiting_time = 0.0
        self.turnaround_time = 0.0
        self.response_time = 0.0
        self.execution_intervals.clear()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Process(pid={self.pid}, arrival={self.arrival_time}, "
            f"burst={self.burst_time}, priority={self.priority}, "
            f"peak_aged_prio={self.peak_effective_priority}, "
            f"rescues={self.starvation_rescues}, "
            f"io_freq={self.io_frequency:.2f})"
        )

    def to_dict(self) -> dict:
        """Serialize the process metrics to a plain dictionary."""
        return {
            "PID": self.pid,
            "Arrival Time": self.arrival_time,
            "Burst Time": self.burst_time,
            "Priority": self.priority,
            "Initial Priority": self.priority,
            "Effective Priority": self.peak_effective_priority,
            "Peak Aged Priority": self.peak_effective_priority,
            "Starvation Rescues": self.starvation_rescues,
            "I/O Frequency": round(self.io_frequency, 3),
            "Start Time": self.start_time,
            "Completion Time": self.completion_time,
            "Waiting Time": round(self.waiting_time, 3),
            "Turnaround Time": round(self.turnaround_time, 3),
            "Response Time": round(self.response_time, 3),
        }
