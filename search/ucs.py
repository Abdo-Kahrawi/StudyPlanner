"""
ucs.py - Uniform Cost Search.
Expands node with lowest g (total sessions) first.
Because every action costs 1, UCS is optimal and equivalent to BFS by depth,
but we keep the priority-queue formulation so cost model can change later.
Guarantees smallest total sessions if a solution exists within caps.
"""
from __future__ import annotations
import heapq, itertools
from typing import Callable, Optional
from .planner_search import StudyPlannerProblem, SearchNode, SearchResult, heuristic_zero

class UniformCostSearch:
    def __init__(self, problem: StudyPlannerProblem):
        self.problem = problem

    def search(self, max_expansions: int = 100_000) -> SearchResult:
        p = self.problem
        start = SearchNode(state=p.start_state(), g=0.0, h=0.0, depth=0)
        frontier = [(start.g, 0, start)]  # (priority=g, tie-breaker, node)
        counter = itertools.count(1)
        visited = {start.state: 0.0}
        expanded = generated = 0
        max_frontier = 1

        while frontier:
            _, _, node = heapq.heappop(frontier)
            if node.g > visited.get(node.state, float("inf")):
                continue
            expanded += 1
            if p.is_goal(node.state):
                return SearchResult(True, node.state, node, p.predicted_scores(node.state),
                                    cost=node.g, nodes_expanded=expanded, nodes_generated=generated,
                                    max_frontier_size=max_frontier, nn_calls=p.nn_calls,
                                    message=f"UCS found optimal plan with {int(node.g)} sessions.")
            if expanded >= max_expansions:
                break
            for a in p.actions(node.state):
                child_state = p.result(node.state, a)
                child_g = node.g + p.step_cost(node.state, a)
                if child_state in visited and visited[child_state] <= child_g:
                    continue
                visited[child_state] = child_g
                child = SearchNode(state=child_state, parent=node, action=a, g=child_g, h=0.0, depth=node.depth+1)
                heapq.heappush(frontier, (child.g, next(counter), child))
                generated += 1
            max_frontier = max(max_frontier, len(frontier))

        best = p.best_effort_state()
        return SearchResult(False, None, None, p.predicted_scores(best), cost=float("inf"),
                            nodes_expanded=expanded, nodes_generated=generated,
                            max_frontier_size=max_frontier, nn_calls=p.nn_calls,
                            message="No solution: target unreachable at this capacity. Best predicted marks: "
                                    f"{p.predicted_scores(best)} vs targets {p.targets}.")

def uniform_cost_search(problem: StudyPlannerProblem, max_expansions: int = 100_000) -> SearchResult:
    return UniformCostSearch(problem).search(max_expansions=max_expansions)
