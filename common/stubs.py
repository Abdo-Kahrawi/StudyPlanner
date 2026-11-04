"""Placeholders so A and B can work in parallel before the real modules exist.

Delete uses of these once nn/ and csp/ are real (target: end of Week 5).
"""
from __future__ import annotations

import numpy as np

from common.types import Plan, ScheduleResult, StudentProfile


class FakePredictor:
    """Monotone, diminishing returns: base mark + gain per session. Not learned from data."""

    name = "fake"

    def __init__(self, base: float = 40.0, gain: float = 6.0, decay: float = 0.85):
        self.base, self.gain, self.decay = base, gain, decay

    def predict(self, profile: StudentProfile, plan: Plan) -> np.ndarray:
        marks = [self.base + self.gain * (1 - self.decay ** n) / (1 - self.decay) for n in plan]
        return np.clip(np.array(marks, dtype=float), 0, 100)


def capacity_schedule(plan: Plan, profile: StudentProfile, **_ignored) -> ScheduleResult:
    """Earliest-deadline-first filling of slots. A quick feasibility stand-in for the real CSP."""
    timetable: dict[tuple[str, int], tuple[int, int]] = {}
    free = [(w, s) for w in range(profile.start_week, max(a.deadline_week for a in profile.assignments))
            for s in range(profile.slots_per_week)]
    for a, n in zip(profile.assignments, plan):
        for k in range(n):
            usable = [ws for ws in free if ws[0] < a.deadline_week]
            if not usable:
                return ScheduleResult(ok=False, timetable={}, log=[f"no slot for {a.id} session {k}"])
            free.remove(usable[0])
            timetable[(a.id, k)] = usable[0]
    return ScheduleResult(ok=True, timetable=timetable)
