"""Shared data types: the contract between Person A and Person B.

Changing anything in this file needs a pull request approved by BOTH people.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

Plan = tuple[int, ...]
"""Sessions per assignment, in assignment order. Example: (4, 5, 3)."""


@dataclass(frozen=True)
class Assignment:
    id: str              # OULAD id_assessment
    kind: str            # "TMA" or "CMA"
    weight: float        # OULAD weight (0-100)
    deadline_week: int   # week index counted from presentation start


@dataclass(frozen=True)
class StudentProfile:
    student_id: int
    history: dict[str, float]              # previous scores, baseline clicks per activity group
    assignments: tuple[Assignment, ...]    # 3 in the minimum version, sorted by deadline
    slots_per_week: int                    # GENERATED: 2, 3 or 4
    start_week: int                        # first week sessions can be placed in
    target: float                          # GENERATED target mark, e.g. 55
    clicks_per_session: int                # GENERATED session length in clicks


class Predictor(Protocol):
    """Implemented by nn.predictor.MarkPredictor and nn.predictor.AverageMarkPredictor."""

    name: str

    def predict(self, profile: StudentProfile, plan: Plan) -> np.ndarray:
        """Predicted mark (0-100) for each assignment, same order as profile.assignments."""
        ...


@dataclass
class SearchResult:
    status: str                            # "found" | "unreachable"
    plan: Plan | None                      # smallest plan meeting the target, if found
    cost: int | None                       # total sessions in plan
    predicted: np.ndarray | None           # predicted marks for plan
    best_plan: Plan | None                 # best schedulable plan when unreachable
    best_predicted: np.ndarray | None
    nodes_expanded: int = 0
    nodes_generated: int = 0
    trace: list[dict] = field(default_factory=list)   # one entry per expansion, for plots/demo


@dataclass
class ScheduleResult:
    ok: bool
    timetable: dict[tuple[str, int], tuple[int, int]]  # (assignment_id, k-th session) -> (week, slot)
    backtracks: int = 0
    assignments: int = 0
    values_pruned: int = 0
    log: list[str] = field(default_factory=list)
