"""Build the scheduling CSP for one plan. Owner: Person B.

Variables  : one per study session, (assignment_id, k) with k = 0..n-1.
Values     : slot indices t = week * slots_per_week + slot, for
             start_week <= week < assignment.deadline_week.
Constraints: (1) one session per slot: t_x != t_y for every pair of sessions;
             (2) sessions of the same assignment are in order: t_(a,k) < t_(a,j) for k < j.
Every pair of variables shares a constraint, so the constraint graph is complete.
"""
from __future__ import annotations

from dataclasses import dataclass

from common.types import Plan, StudentProfile

Var = tuple[str, int]


@dataclass
class SchedulingCSP:
    variables: list[Var]
    domains: dict[Var, list[int]]
    slots_per_week: int

    def consistent(self, x: Var, a: int, y: Var, b: int) -> bool:
        """True if x=a and y=b satisfy every constraint between x and y."""
        if x[0] == y[0]:                     # same assignment: keep the order
            return a < b if x[1] < y[1] else a > b
        return a != b                        # different assignments: different slots

    def neighbors(self, x: Var) -> list[Var]:
        return [y for y in self.variables if y != x]

    def decode(self, t: int) -> tuple[int, int]:
        return divmod(t, self.slots_per_week)


def build_csp(plan: Plan, profile: StudentProfile) -> SchedulingCSP:
    if len(plan) != len(profile.assignments):
        raise ValueError(f"plan has {len(plan)} entries but profile has {len(profile.assignments)} assignments")
    s = profile.slots_per_week
    variables: list[Var] = []
    domains: dict[Var, list[int]] = {}
    for a, n in zip(profile.assignments, plan):
        values = [w * s + k for w in range(profile.start_week, a.deadline_week) for k in range(s)]
        for i in range(n):
            v = (a.id, i)
            variables.append(v)
            domains[v] = list(values)
    return SchedulingCSP(variables, domains, s)
