"""
planner_search.py - Search formulation for P1.

State: tuple[int, ...] = sessions per assignment, e.g. (4,2,3) for 3 assignments.
  Start: (0,0,...,0)
  Action: add 1 session to assignment i  (one action per assignment)
  Cost g(n): total sessions = sum(state)  (every session costs 1, so UCS == BFS by depth)
  Goal: predicted_score_i(state) >= target_i for ALL i
        AND (optionally) CSP-feasible - checked by caller / integration layer.
        Search itself does NOT call CSP; integration does: if CSP rejects a goal
        node, integration discards it and continues search (see run_demo.py later).

Predictor interface:
  predictor.predict_score(student_profile_i, proposed_plan) -> float 0-100
  We wrap it: predicted_scores(state) = [predictor_i(state_i) for each assignment]
  To avoid re-calling NN for every node, we cache scores per (assignment, sessions).

Heuristics (all admissible-ish, documented for report):
  h_zero = 0  -> UCS behaviour
  h_max_deficit = max_i deficit_i  where deficit_i = max(0, target_i - pred_i)
    Intuition: 1 session cannot raise a mark by more than GAIN_PER_SESSION (we use
    optimistic gain, see below). So h = ceil(max_deficit / max_gain_per_session)
  h_remaining_deficit = sum_i ceil(deficit_i / max_gain_per_session)
    Stronger, still admissible IF max_gain is a true upper bound on per-session gain.
    We estimate max_gain from data / config; default 5.0 marks per session (tunable).
    If NN is non-monotonic, heuristic can overestimate - report must note this and
    monotonicity test in nn/ is the guard.

Counters required by spec:
  nodes_expanded, nodes_generated, max_frontier, nn_calls

Also provides a tiny brute-force checker for tests on small example with known answer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional, Dict
import math

State = Tuple[int, ...]

# --- Heuristics ---
def heuristic_zero(problem, state: State) -> float:
    return 0.0

def _deficits(problem, state: State) -> List[float]:
    preds = problem.predicted_scores(state)
    return [max(0.0, t - p) for t, p in zip(problem.targets, preds)]

def heuristic_max_deficit(problem, state: State) -> float:
    """Admissible if gain_per_session is an upper bound on true gain."""
    deficits = _deficits(problem, state)
    if not deficits or max(deficits) <= 0:
        return 0.0
    return math.ceil(max(deficits) / problem.max_gain_per_session)

def heuristic_remaining_deficit(problem, state: State) -> float:
    """Sum of per-assignment needed sessions (stronger)."""
    deficits = _deficits(problem, state)
    return float(sum(math.ceil(d / problem.max_gain_per_session) for d in deficits))

# alias used in papers
heuristic_sum_deficit = heuristic_remaining_deficit

@dataclass
class SearchNode:
    state: State
    parent: Optional["SearchNode"] = None
    action: Optional[int] = None  # assignment index we added a session to, None for root
    g: float = 0.0  # path cost = total sessions
    h: float = 0.0  # heuristic
    depth: int = 0

    @property
    def f(self) -> float:
        return self.g + self.h

    def path(self) -> List["SearchNode"]:
        node, out = self, []
        while node is not None:
            out.append(node)
            node = node.parent
        return list(reversed(out))

    def actions(self) -> List[int]:
        """Actions that led here, in order."""
        return [n.action for n in self.path()[1:] if n.action is not None]

@dataclass
class SearchResult:
    success: bool
    solution_state: Optional[State] = None
    solution_node: Optional[SearchNode] = None
    predicted_scores: List[float] = field(default_factory=list)
    cost: float = float("inf")  # total sessions
    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier_size: int = 0
    nn_calls: int = 0
    message: str = ""

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "allocation": list(self.solution_state) if self.solution_state else None,
            "predicted_scores": self.predicted_scores,
            "cost_total_sessions": self.cost,
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "max_frontier_size": self.max_frontier_size,
            "nn_calls": self.nn_calls,
            "message": self.message,
        }

class StudyPlannerProblem:
    """
    Structured input for all search algorithms.
    student_profiles: list[dict] one per assignment, each is the nn feature dict
                      (see nn/effort.py FEATURE_ORDER) for that assignment.
    targets: list[float] target mark per assignment, e.g. [55,55,55]
    predictor: object with predict_score(profile, proposed_plan:int)->float
               OR callable (profile, sessions)->float. If None, a dummy
               linear predictor can be injected for unit tests.
    max_sessions_per_assignment: cap to keep search finite, e.g. 10
    max_total_sessions: global cap, e.g. 20
    max_gain_per_session: optimistic marks gained per session, for heuristic.
    """
    def __init__(
        self,
        student_profiles: List[dict],
        targets: List[float],
        predictor=None,
        max_sessions_per_assignment: int = 10,
        max_total_sessions: int = 30,
        max_gain_per_session: float = 5.0,
    ):
        assert len(student_profiles) == len(targets), "profiles and targets must align"
        self.profiles = student_profiles
        self.targets = list(map(float, targets))
        self.n = len(targets)
        self.predictor = predictor
        self.max_per = int(max_sessions_per_assignment)
        self.max_total = int(max_total_sessions)
        self.max_gain_per_session = float(max_gain_per_session)
        self._cache: Dict[Tuple[int, int], float] = {}
        self.nn_calls = 0

    def start_state(self) -> State:
        return tuple(0 for _ in range(self.n))

    def is_goal(self, state: State) -> bool:
        return all(p >= t for p, t in zip(self.predicted_scores(state), self.targets))

    def actions(self, state: State) -> List[int]:
        """Add 1 session to any assignment not yet at cap and not over total cap."""
        if sum(state) >= self.max_total:
            return []
        return [i for i in range(self.n) if state[i] < self.max_per]

    def result(self, state: State, action: int) -> State:
        lst = list(state); lst[action] += 1
        return tuple(lst)

    def step_cost(self, state: State, action: int) -> float:
        return 1.0  # every session costs 1

    def path_cost(self, state: State) -> float:
        return float(sum(state))

    def _predict_one(self, assignment_idx: int, sessions: int) -> float:
        key = (assignment_idx, sessions)
        if key in self._cache:
            return self._cache[key]
        profile = self.profiles[assignment_idx]
        if self.predictor is None:
            # Fallback for tests without NN: base = cumulative_avg or prev_score, +2 per session
            base = profile.get("cumulative_avg_score") or profile.get("prev_score") or 40.0
            pred = float(base) + 2.0 * sessions
        elif hasattr(self.predictor, "predict_score"):
            pred = float(self.predictor.predict_score(profile, sessions))
        else:
            pred = float(self.predictor(profile, sessions))
        self.nn_calls += 1
        self._cache[key] = max(0.0, min(100.0, pred))
        return self._cache[key]

    def predicted_scores(self, state: State) -> List[float]:
        return [self._predict_one(i, s) for i, s in enumerate(state)]

    def best_effort_state(self) -> State:
        """Best state reachable at caps (for 'no solution' message)."""
        return tuple(self.max_per for _ in range(self.n))
