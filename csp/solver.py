"""Entry point used by integration/. Owner: Person B.

Variables  : one per session, (assignment_id, k).
Domains    : (week, slot) pairs with start_week <= week < assignment.deadline_week.
Constraints: one session per slot (all-different);
             sessions of the same assignment in order: (a, k) before (a, k+1).
Switches   : use_fc (Forward Checking), use_ac3 (AC-3 before search), use_mac (AC-3 after
             every assignment, extension). Backtracking is always on.
Counters   : backtracks, assignments, values_pruned (plus fc_removed, ac3_removed, revisions).
"""
from __future__ import annotations

from common.types import Plan, ScheduleResult, StudentProfile
from csp.ac3 import ac3
from csp.backtracking import backtracking_search
from csp.variables import build_csp

MODES = {"bt": (False, False), "fc": (True, False), "ac3": (True, True)}
"""Experiment 2 modes: name -> (use_fc, use_ac3)."""


def capacity_ok(plan: Plan, profile: StudentProfile) -> bool:
    """Counting check: for each deadline, sessions due by then must fit in the slots before it."""
    due = 0
    for a, n in sorted(zip(profile.assignments, plan), key=lambda p: p[0].deadline_week):
        due += n
        if due > max(0, a.deadline_week - profile.start_week) * profile.slots_per_week:
            return False
    return True


def schedule(plan: Plan, profile: StudentProfile, use_fc: bool = True,
             use_ac3: bool = True, use_mac: bool = False,
             precheck: bool = False, max_assignments: int = 200_000) -> ScheduleResult:
    """Place every session of the plan into a weekly slot, or report that it cannot be done.

    precheck=True runs the counting check first (used inside the planner for speed; off in
    Experiment 2 so the three modes are compared on the same work).
    """
    stats = {"assignments": 0, "backtracks": 0, "values_pruned": 0,
             "fc_removed": 0, "ac3_removed": 0, "revisions": 0}
    log: list[str] = []

    def result(ok: bool, timetable=None) -> ScheduleResult:
        r = ScheduleResult(ok=ok, timetable=timetable or {}, backtracks=stats["backtracks"],
                           assignments=stats["assignments"], values_pruned=stats["values_pruned"],
                           log=log)
        r.stats = stats            # extra counters for plots; not part of the contract
        return r

    if sum(plan) == 0:
        return result(True)
    if precheck and not capacity_ok(plan, profile):
        log.append("counting check failed: more sessions than slots before a deadline")
        return result(False)

    csp = build_csp(plan, profile)
    domains = {v: list(d) for v, d in csp.domains.items()}
    if any(not d for d in domains.values()):
        log.append("empty domain: a deadline is at or before the first plannable week")
        return result(False)
    if use_ac3:
        before = sum(len(d) for d in domains.values())
        if not ac3(csp, domains, stats):
            log.append("AC-3 emptied a domain before search")
            return result(False)
        log.append(f"AC-3 removed {before - sum(len(d) for d in domains.values())} values before search")

    solution = backtracking_search(csp, domains, stats, use_fc=use_fc, use_mac=use_mac,
                                   max_assignments=max_assignments, log=log)
    if solution is None:
        return result(False)
    return result(True, {var: csp.decode(t) for var, t in solution.items()})
