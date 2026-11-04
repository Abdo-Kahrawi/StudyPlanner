"""Entry point used by integration/. Owner: Person B.

Variables  : one per session, (assignment_id, k).
Domains    : (week, slot) pairs with start_week <= week < assignment.deadline_week.
Constraints: one session per slot (all-different);
             sessions of the same assignment in order: (a, k) before (a, k+1).
Switches   : use_fc (Forward Checking), use_ac3 (AC-3 before search), use_mac (extension).
Counters   : backtracks, assignments, values_pruned.
"""
from common.types import Plan, ScheduleResult, StudentProfile


def schedule(plan: Plan, profile: StudentProfile, use_fc: bool = True,
             use_ac3: bool = True, use_mac: bool = False) -> ScheduleResult:
    raise NotImplementedError("Person B, Week 2 (BT), Week 3 (FC, AC-3)")
