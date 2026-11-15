"""Backtracking search with optional Forward Checking and MAC. Owner: Person B.

Variables are tried in a fixed order (assignments by deadline, sessions in order) and
values from earliest slot to latest, so BT, BT+FC and BT+FC+AC-3 differ only in pruning.
"""
from __future__ import annotations

from csp.ac3 import ac3
from csp.forward_checking import forward_check
from csp.variables import SchedulingCSP, Var


class LimitReached(Exception):
    pass


def backtracking_search(csp: SchedulingCSP, domains: dict[Var, list[int]], stats: dict,
                        use_fc: bool, use_mac: bool, max_assignments: int,
                        log: list[str]) -> dict[Var, int] | None:
    order = list(csp.variables)
    assignment: dict[Var, int] = {}

    def consistent_with_assignment(var: Var, value: int) -> bool:
        return all(csp.consistent(var, value, y, b) for y, b in assignment.items())

    def recurse(i: int, doms: dict[Var, list[int]]) -> bool:
        if i == len(order):
            return True
        var = order[i]
        for value in doms[var]:
            if not use_fc and not use_mac and not consistent_with_assignment(var, value):
                continue
            stats["assignments"] += 1
            if stats["assignments"] > max_assignments:
                raise LimitReached
            assignment[var] = value
            new = {v: list(d) for v, d in doms.items()}
            new[var] = [value]
            ok = True
            if use_fc:
                ok = forward_check(csp, new, var, value, order[i + 1:], stats)
            if ok and use_mac:
                ok = ac3(csp, new, stats, queue=[(y, var) for y in order[i + 1:]])
            if ok and recurse(i + 1, new):
                return True
            del assignment[var]
            stats["backtracks"] += 1
            if len(log) < 200:
                log.append(f"backtrack: {var} != slot {value}")
        return False

    try:
        if recurse(0, domains):
            return dict(assignment)
    except LimitReached:
        log.append(f"stopped: more than {max_assignments} assignments")
    return None
