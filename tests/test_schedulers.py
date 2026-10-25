"""
tests/test_schedulers.py
------------------------
Unit tests for CPU schedulers, Priority Aging, Starvation Watchdog, and Dual-Core SMP.
Compatible with standard library unittest.
"""

import unittest
from core.process import Process
from core.schedulers import fcfs, priority_aging
from core.simulation_engine import SimulationEngine
from scheduler.adaptive_scheduler import TrueAdaptiveScheduler, DualCoreAdaptiveScheduler


class TestSchedulers(unittest.TestCase):

    def test_fcfs(self):
        procs = [
            Process(pid=1, arrival_time=0, burst_time=5),
            Process(pid=2, arrival_time=2, burst_time=3),
        ]
        res = fcfs(procs)
        self.assertEqual(res["processes"][0].completion_time, 5)
        self.assertEqual(res["processes"][1].completion_time, 8)
        self.assertEqual(res["processes"][0].waiting_time, 0.0)
        self.assertEqual(res["processes"][1].waiting_time, 3.0)

    def test_priority_aging_prevents_starvation(self):
        procs = [
            Process(pid=1, arrival_time=0, burst_time=10, priority=5),
            Process(pid=2, arrival_time=1, burst_time=2, priority=1),
        ]
        res_aging = priority_aging(procs, aging_interval=2)
        self.assertTrue(res_aging["algorithm"].startswith("Priority (Aging"))
        self.assertLessEqual(res_aging["processes"][0].effective_priority, res_aging["processes"][0].priority)

    def test_anti_starvation_watchdog(self):
        # P1 is low priority, followed by a continuous stream of high-priority short tasks
        procs = [
            Process(pid=1, arrival_time=0, burst_time=20, priority=9),
            Process(pid=2, arrival_time=0, burst_time=4, priority=1),
            Process(pid=3, arrival_time=4, burst_time=4, priority=1),
            Process(pid=4, arrival_time=8, burst_time=4, priority=1),
            Process(pid=5, arrival_time=12, burst_time=4, priority=1),
        ]
        sched = TrueAdaptiveScheduler(starvation_threshold=8, base_quantum=3)
        res = sched.run(procs)
        # P1 waited 8 ticks while P2 and P3 executed -> Watchdog must trigger rescue
        self.assertGreaterEqual(res["starvation_rescues"], 1)
        p1 = next(p for p in res["processes"] if p.pid == 1)
        self.assertIsNotNone(p1.completion_time)

    def test_dual_core_parallel_speedup(self):
        procs = [
            Process(pid=1, arrival_time=0, burst_time=10, priority=3),
            Process(pid=2, arrival_time=0, burst_time=10, priority=3),
        ]
        single = TrueAdaptiveScheduler().run(procs)
        single_makespan = max(p.completion_time for p in single["processes"])

        dual = DualCoreAdaptiveScheduler().run(procs)
        dual_makespan = dual["makespan"]

        self.assertEqual(dual_makespan, 10)
        self.assertEqual(single_makespan, 20)
        self.assertLess(dual_makespan, single_makespan)


if __name__ == "__main__":
    unittest.main()
