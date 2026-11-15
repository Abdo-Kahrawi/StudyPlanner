"""
greedy.py - Greedy Best-First Search.
Expands node with lowest h only (ignores g / cost so far).
Fast, but NOT optimal: can return a plan with more sessions than necessary,
and can get stuck exploring a promising-looking branch.
Used as baseline in Experiment (1): UCS vs Greedy vs A*.
"""
from __future__ import annotations
import heapq, itertools
from .planner_search import StudyPlannerProblem, SearchNode, SearchResult, heuristic_remaining_deficit

class GreedyBestFirstSearch:
    def __init__(self, problem: StudyPlannerProblem, heuristic=None):
        self.problem = problem
        self.heuristic = heuristic or heuristic_remaining_deficit

    def search(self, max_expansions: int = 100_000) -> SearchResult:
        p = self.problem
        h0 = self.heuristic(p, p.start_state())
        start = SearchNode(state=p.start_state(), g=0.0, h=h0, depth=0)
        frontier = [(start.h, 0, start)]  # priority = h ONLY
        counter = itertools.count(1)
        visited = {start.state}
        expanded = generated = 0
        max_frontier = 1

        while frontier:
            _, _, node = heapq.heappop(frontier)
            expanded += 1
            if p.is_goal(node.state):
                return SearchResult(True, node.state, node, p.predicted_scores(node.state),
                                    cost=node.g, nodes_expanded=expanded, nodes_generated=generated,
                                    max_frontier_size=max_frontier, nn_calls=p.nn_calls,
                                    message=f"Greedy found plan with {int(node.g)} sessions (not guaranteed optimal).")
            if expanded >= max_expansions:
                break
            for a in p.actions(node.state):
                child_state = p.result(node.state, a)
                if child_state in visited:
                    continue
                visited.add(child_state)
                h = self.heuristic(p, child_state)
                child = SearchNode(state=child_state, parent=node, action=a,
                                   g=node.g + p.step_cost(node.state, a), h=h, depth=node.depth+1)
                heapq.heappush(frontier, (child.h, next(counter), child))
                generated += 1
            max_frontier = max(max_frontier, len(frontier))

        best = p.best_effort_state()
        return SearchResult(False, None, None, p.predicted_scores(best), cost=float("inf"),
                            nodes_expanded=expanded, nodes_generated=generated,
                            max_frontier_size=max_frontier, nn_calls=p.nn_calls,
                            message="No solution: target unreachable at this capacity. "
                                    f"Best predicted marks: {p.predicted_scores(best)} vs targets {p.targets}.")

def greedy_search(problem: StudyPlannerProblem, heuristic=None, max_expansions: int = 100_000) -> SearchResult:
    return GreedyBestFirstSearch(problem, heuristic=heuristic).search(max_expansions=max_expansions)
