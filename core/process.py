"""
core/process.py
---------------
Defines the Process class representing a single schedulable unit.
Tracks all scheduling metrics and execution intervals for Gantt chart rendering.
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

    # --- Derived scheduling metrics (populated after simulation) ---
    waiting_time: float = field(default=0.0, init=False)
    turnaround_time: float = field(default=0.0, init=False)
    response_time: float = field(default=0.0, init=False)

    # --- Execution history for Gantt chart rendering ---
    # Each tuple is (start_tick, end_tick) on the CPU.
    execution_intervals: List[Tuple[int, int]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.remaining_time = self.burst_time

    # ------------------------------------------------------------------
    # Execution history helpers
    # ------------------------------------------------------------------

    def log_interval(self, start: int, end: int) -> None:
        """Record a CPU execution interval (start, end) for Gantt rendering."""
        if start >= end:
            return
        self.execution_intervals.append((start, end))

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
            f"io_freq={self.io_frequency:.2f})"
        )

    def to_dict(self) -> dict:
        """Serialize the process metrics to a plain dictionary."""
        return {
            "PID": self.pid,
            "Arrival Time": self.arrival_time,
            "Burst Time": self.burst_time,
            "Priority": self.priority,
            "I/O Frequency": round(self.io_frequency, 3),
            "Start Time": self.start_time,
            "Completion Time": self.completion_time,
            "Waiting Time": round(self.waiting_time, 3),
            "Turnaround Time": round(self.turnaround_time, 3),
            "Response Time": round(self.response_time, 3),
        }
