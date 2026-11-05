"""Search proposes, NN checks marks, CSP checks the schedule. Owner: Person B.

Loop:
    1. search() pops a plan.
    2. predictor.predict(plan) >= target for every assignment?
    3. schedule(plan) ok?  (results cached per plan)
    4. Both yes -> done. NN yes but CSP no -> record an "NN yes, CSP no" event, continue.
    5. Nothing left -> "No solution: target unreachable at this capacity" + best predicted marks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from common import config
from common.types import Plan, Predictor, ScheduleResult, SearchResult, StudentProfile
from csp.solver import schedule

NO_SOLUTION = "No solution: target unreachable at this capacity."


@dataclass
class CSPCall:
    plan: Plan
    nn_ok: bool
    ok: bool
    assignments: int
    backtracks: int
    values_pruned: int


@dataclass
class PlannerOutput:
    profile: StudentProfile
    search: SearchResult
    schedule: ScheduleResult | None
    nn_yes_csp_no: list[Plan] = field(default_factory=list)
    csp_calls: list[CSPCall] = field(default_factory=list)
    message: str = ""
    seconds: float = 0.0

    @property
    def found(self) -> bool:
        return self.search.status == "found"

    @property
    def plan(self) -> Plan | None:
        return self.search.plan if self.found else self.search.best_plan

    @property
    def predicted(self) -> np.ndarray | None:
        return self.search.predicted if self.found else self.search.best_predicted


def _default_search() -> Callable:
    from search import search
    return search


def plan_for(profile: StudentProfile, predictor: Predictor, strategy: str = "ucs",
             use_fc: bool = True, use_ac3: bool = True, use_mac: bool = False,
             search_fn: Callable | None = None,
             max_sessions: int = config.MAX_SESSIONS_PER_ASSIGNMENT) -> PlannerOutput:
    """Run the full planner for one student. search_fn defaults to search.search (Person A)."""
    search_fn = search_fn or _default_search()
    cache: dict[Plan, ScheduleResult] = {}
    calls: list[CSPCall] = []
    nn_yes_csp_no: list[Plan] = []

    def is_schedulable(plan: Plan) -> bool:
        plan = tuple(int(x) for x in plan)
        if plan not in cache:
            res = schedule(plan, profile, use_fc=use_fc, use_ac3=use_ac3, use_mac=use_mac,
                           precheck=True)
            cache[plan] = res
            nn_ok = bool(np.all(predictor.predict(profile, plan) >= profile.target))
            calls.append(CSPCall(plan, nn_ok, res.ok, res.assignments, res.backtracks,
                                 res.values_pruned))
            if nn_ok and not res.ok:
                nn_yes_csp_no.append(plan)
        return cache[plan].ok

    t0 = time.perf_counter()
    result = search_fn(profile, predictor, strategy, is_schedulable, max_sessions)
    out = PlannerOutput(profile, result, None, nn_yes_csp_no, calls)

    plan = out.plan
    if plan is not None:
        is_schedulable(plan)
        out.schedule = cache[tuple(int(x) for x in plan)]
    marks = out.predicted
    marks_txt = ", ".join(f"{a.id}: {m:.1f}" for a, m in zip(profile.assignments, marks)) \
        if marks is not None else "n/a"
    if out.found:
        out.message = f"Plan found: {plan} ({sum(plan)} sessions). Predicted marks: {marks_txt}."
    else:
        out.message = f"{NO_SOLUTION} Best predicted marks: {marks_txt}" + \
                      (f" with plan {plan}." if plan is not None else ".")
    out.seconds = time.perf_counter() - t0
    return out


def format_timetable(out: PlannerOutput) -> str:
    """Week-by-week text timetable: one line per week that has sessions."""
    if out.schedule is None or not out.schedule.timetable:
        return "(no sessions scheduled)"
    by_week: dict[int, list[str]] = {}
    for (aid, k), (week, slot) in sorted(out.schedule.timetable.items(), key=lambda kv: kv[1]):
        by_week.setdefault(week, []).append(f"slot {slot + 1}: {aid} #{k + 1}")
    return "\n".join(f"Week {w + 1:>2}: " + ", ".join(items) for w, items in sorted(by_week.items()))
