"""
astar.py - A* Search (optional extension in spec, required for Experiment 1).
Expands node with lowest f = g + h.
Optimal IF h never overestimates true remaining cost (admissible).
Our h = ceil(deficit / max_gain_per_session) is admissible when max_gain
is a true upper bound on NN gain per session. Check monotonicity test first.
If h overestimates, A* may return suboptimal plan - note in report.
"""
from __future__ import annotations
import heapq, itertools
from .planner_search import StudyPlannerProblem, SearchNode, SearchResult, heuristic_remaining_deficit

class AStarSearch:
    def __init__(self, problem: StudyPlannerProblem, heuristic=None):
        self.problem = problem
        self.heuristic = heuristic or heuristic_remaining_deficit

    def search(self, max_expansions: int = 100_000) -> SearchResult:
        p = self.problem
        h0 = self.heuristic(p, p.start_state())
        start = SearchNode(state=p.start_state(), g=0.0, h=h0, depth=0)
        frontier = [(start.f, start.h, 0, start)]  # (f, h tie-break, counter, node)
        counter = itertools.count(1)
        visited = {start.state: 0.0}
        expanded = generated = 0
        max_frontier = 1

        while frontier:
            _, _, _, node = heapq.heappop(frontier)
            if node.g > visited.get(node.state, float("inf")):
                continue
            expanded += 1
            if p.is_goal(node.state):
                return SearchResult(True, node.state, node, p.predicted_scores(node.state),
                                    cost=node.g, nodes_expanded=expanded, nodes_generated=generated,
                                    max_frontier_size=max_frontier, nn_calls=p.nn_calls,
                                    message=f"A* found optimal plan with {int(node.g)} sessions (if h admissible).")
            if expanded >= max_expansions:
                break
            for a in p.actions(node.state):
                child_state = p.result(node.state, a)
                child_g = node.g + p.step_cost(node.state, a)
                if child_state in visited and visited[child_state] <= child_g:
                    continue
                visited[child_state] = child_g
                h = self.heuristic(p, child_state)
                child = SearchNode(state=child_state, parent=node, action=a, g=child_g, h=h, depth=node.depth+1)
                heapq.heappush(frontier, (child.f, child.h, next(counter), child))
                generated += 1
            max_frontier = max(max_frontier, len(frontier))

        best = p.best_effort_state()
        return SearchResult(False, None, None, p.predicted_scores(best), cost=float("inf"),
                            nodes_expanded=expanded, nodes_generated=generated,
                            max_frontier_size=max_frontier, nn_calls=p.nn_calls,
                            message="No solution: target unreachable at this capacity. "
                                    f"Best predicted marks: {p.predicted_scores(best)} vs targets {p.targets}.")

def astar_search(problem: StudyPlannerProblem, heuristic=None, max_expansions: int = 100_000) -> SearchResult:
    return AStarSearch(problem, heuristic=heuristic).search(max_expansions=max_expansions)
