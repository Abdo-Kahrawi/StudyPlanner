"""search() entry point used by integration/ and eval/. Bridges Abdo's search classes
(UCS, Greedy, A* over StudyPlannerProblem) to the shared contract in common/types.py.

    search(profile, predictor, strategy, is_schedulable, max_sessions) -> common.types.SearchResult

- State = plan = sessions per assignment, start (0, ..., 0), each action adds one session.
- Marks come from predictor.predict(profile, plan) (common.types.Predictor). Each
  assignment's mark depends only on its own sessions, so marks are cached per
  (assignment, sessions) by StudyPlannerProblem.
- Goal = every predicted mark >= profile.target AND is_schedulable(plan) (the CSP).
  If the NN says yes but the CSP says no, the search keeps going.
- Heuristic for Greedy / A*: max_gain_per_session is set to the largest one-session gain
  the predictor gives for this profile, so the deficit heuristic never overestimates and
  A* stays optimal.
- If no goal is found, best_plan is the schedulable plan with the best worst-case margin
  (min over assignments of mark - target) among the plans the search expanded.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from common.types import Plan, SearchResult, StudentProfile
from .astar import astar_search
from .greedy import greedy_search
from .planner_search import StudyPlannerProblem
from .ucs import uniform_cost_search

STRATEGIES = {"ucs": uniform_cost_search, "greedy": greedy_search, "astar": astar_search}


class _ProfileProblem(StudyPlannerProblem):
    """StudyPlannerProblem driven by a common.types.Predictor and a CSP check."""

    def __init__(self, profile: StudentProfile, predictor, is_schedulable: Callable[[Plan], bool],
                 max_sessions: int):
        n = len(profile.assignments)
        self.profile = profile
        self.contract_predictor = predictor
        self.is_schedulable = is_schedulable
        super().__init__(student_profiles=[{} for _ in range(n)], targets=[profile.target] * n,
                         predictor=None, max_sessions_per_assignment=max_sessions,
                         max_total_sessions=n * max_sessions)
        self.max_gain_per_session = self._max_gain()
        self.trace: list[dict] = []
        self.best: tuple[float, Plan, list[float]] | None = None

    def _predict_one(self, assignment_idx: int, sessions: int) -> float:
        key = (assignment_idx, sessions)
        if key not in self._cache:
            plan = [0] * self.n
            plan[assignment_idx] = sessions
            marks = self.contract_predictor.predict(self.profile, tuple(plan))
            self.nn_calls += 1
            self._cache[key] = float(np.clip(marks[assignment_idx], 0.0, 100.0))
        return self._cache[key]

    def _max_gain(self) -> float:
        gains = [self._predict_one(i, s + 1) - self._predict_one(i, s)
                 for i in range(self.n) for s in range(self.max_per)]
        return max(max(gains, default=0.0), 1e-6)

    def is_goal(self, state) -> bool:
        """Called once per expanded node by UCS, Greedy and A*: also records the trace."""
        plan = tuple(int(x) for x in state)
        marks = self.predicted_scores(plan)
        margin = min(m - t for m, t in zip(marks, self.targets))
        self.trace.append({"step": len(self.trace) + 1, "plan": plan, "cost": sum(plan),
                           "min_margin": margin})
        if margin >= 0:
            return self.is_schedulable(plan)
        if (self.best is None or margin > self.best[0]) and self.is_schedulable(plan):
            self.best = (margin, plan, marks)
        return False


def search(profile: StudentProfile, predictor, strategy: str = "ucs",
           is_schedulable: Callable[[Plan], bool] = lambda plan: True,
           max_sessions: int = 10) -> SearchResult:
    """Smallest plan (fewest sessions) whose predicted marks meet the target and that fits."""
    try:
        algorithm = STRATEGIES[strategy.lower().replace("*", "star")]
    except KeyError:
        raise ValueError(f"unknown strategy {strategy!r}; use one of {sorted(STRATEGIES)}") from None
    problem = _ProfileProblem(profile, predictor, is_schedulable, max_sessions)
    res = algorithm(problem)
    if res.success:
        plan = tuple(int(x) for x in res.solution_state)
        marks = np.array(res.predicted_scores, dtype=float)
        return SearchResult("found", plan, sum(plan), marks, plan, marks,
                            nodes_expanded=res.nodes_expanded, nodes_generated=res.nodes_generated,
                            trace=problem.trace)
    best_plan = best_marks = None
    if problem.best is not None:
        _, best_plan, marks = problem.best
        best_marks = np.array(marks, dtype=float)
    return SearchResult("unreachable", None, None, None, best_plan, best_marks,
                        nodes_expanded=res.nodes_expanded, nodes_generated=res.nodes_generated,
                        trace=problem.trace)
