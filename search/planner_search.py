"""Entry point used by integration/. Owner: Person A.

State   : Plan, tuple of sessions per assignment, starting at (0, 0, 0).
Action  : add one session to one assignment.
Cost    : 1 per session (UCS minimises total sessions).
Goal    : every predicted mark >= target AND is_schedulable(plan).
Greedy  : h = total shortfall below target (or sessions still needed).
A*      : f = g + h with a monotone (consistent) h. Extension.
Counters: nodes_expanded, nodes_generated; append one dict per expansion to trace.
"""
from typing import Callable

from common.types import Plan, Predictor, SearchResult, StudentProfile

STRATEGIES = ("ucs", "greedy", "astar")


def search(profile: StudentProfile, predictor: Predictor, strategy: str,
           is_schedulable: Callable[[Plan], bool], max_sessions: int) -> SearchResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    raise NotImplementedError("Person A, Week 4 (ucs, greedy), Week 5 (astar)")
