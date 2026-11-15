"""AC-3 arc consistency for the scheduling CSP. Owner: Person B."""
from __future__ import annotations

from collections import deque

from csp.variables import SchedulingCSP, Var


def revise(csp: SchedulingCSP, domains: dict[Var, list[int]], x: Var, y: Var, stats: dict) -> bool:
    """Remove values of x that have no support in y. Returns True if x's domain changed."""
    keep = [a for a in domains[x] if any(csp.consistent(x, a, y, b) for b in domains[y])]
    removed = len(domains[x]) - len(keep)
    if removed:
        domains[x] = keep
        stats["values_pruned"] += removed
        stats["ac3_removed"] += removed
    stats["revisions"] += 1
    return removed > 0


def ac3(csp: SchedulingCSP, domains: dict[Var, list[int]], stats: dict,
        queue: list[tuple[Var, Var]] | None = None) -> bool:
    """Make every arc consistent. Returns False if some domain becomes empty.

    queue=None starts from all arcs (preprocessing). MAC passes only the arcs into
    the variable it just assigned.
    """
    q = deque(queue if queue is not None else
              [(x, y) for x in csp.variables for y in csp.neighbors(x)])
    while q:
        x, y = q.popleft()
        if revise(csp, domains, x, y, stats):
            if not domains[x]:
                return False
            for z in csp.neighbors(x):
                if z != y:
                    q.append((z, x))
    return True
