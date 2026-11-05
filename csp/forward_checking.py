"""Forward Checking for the scheduling CSP. Owner: Person B."""
from __future__ import annotations

from csp.variables import SchedulingCSP, Var


def forward_check(csp: SchedulingCSP, domains: dict[Var, list[int]], var: Var, value: int,
                  unassigned: list[Var], stats: dict) -> bool:
    """After var=value, remove now-impossible values from unassigned neighbours.

    Mutates domains. Returns False if some neighbour has no value left (dead end).
    """
    for y in unassigned:
        keep = [b for b in domains[y] if csp.consistent(var, value, y, b)]
        removed = len(domains[y]) - len(keep)
        if removed:
            domains[y] = keep
            stats["values_pruned"] += removed
            stats["fc_removed"] += removed
            if not keep:
                return False
    return True
