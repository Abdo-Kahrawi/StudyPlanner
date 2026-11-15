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


class HistoryFakePredictor(FakePredictor):
    """FakePredictor whose base mark is the student's previous mean minus 15 (default 40)."""

    name = "fake-history"

    def predict(self, profile: StudentProfile, plan: Plan) -> np.ndarray:
        self.base = profile.history.get("prev_score_mean", 55.0) - 15.0
        return super().predict(profile, plan)


def reference_search(profile, predictor, strategy, is_schedulable, max_sessions):
    """Brute-force stand-in for search.search: tries plans in order of total sessions.

    Same signature and SearchResult as the real search, so integration/ can be built and
    tested before Person A's UCS/Greedy/A* land. Ignores `strategy`. Not for experiments.
    """
    from itertools import product

    from common.types import SearchResult

    n = len(profile.assignments)
    plans = sorted(product(range(max_sessions + 1), repeat=n), key=lambda p: (sum(p), p))
    best, best_marks, best_key = None, None, None
    trace, expanded = [], 0
    for plan in plans:
        expanded += 1
        marks = predictor.predict(profile, plan)
        key = float(np.min(marks - profile.target))
        meets = bool(np.all(marks >= profile.target))
        trace.append({"step": expanded, "plan": plan, "cost": sum(plan), "min_margin": key})
        if (best_key is None or key > best_key) and is_schedulable(plan):
            best, best_marks, best_key = plan, marks, key
        if meets and is_schedulable(plan):
            return SearchResult("found", plan, sum(plan), marks, plan, marks,
                                nodes_expanded=expanded, nodes_generated=expanded, trace=trace)
    return SearchResult("unreachable", None, None, None, best, best_marks,
                        nodes_expanded=expanded, nodes_generated=expanded, trace=trace)
