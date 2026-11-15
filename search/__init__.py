"""
search package for P1 Assessment-Aware Study Planner.
Exposes search classes so other modules do:
  from search import StudyPlannerProblem, uniform_cost_search, greedy_search, astar_search
  from search.planner_search import SearchNode, SearchResult
and the planner / experiments use the contract entry point:
  from search import search    # search(profile, predictor, strategy, is_schedulable, max_sessions)
"""
from .planner_search import (
    StudyPlannerProblem,
    SearchNode,
    SearchResult,
    heuristic_remaining_deficit,
    heuristic_max_deficit,
    heuristic_zero,
)
from .ucs import uniform_cost_search, UniformCostSearch
from .greedy import greedy_search, GreedyBestFirstSearch
from .astar import astar_search, AStarSearch
from .adapter import search

__all__ = [
    "search",
    "StudyPlannerProblem", "SearchNode", "SearchResult",
    "heuristic_remaining_deficit", "heuristic_max_deficit", "heuristic_zero",
    "uniform_cost_search", "UniformCostSearch",
    "greedy_search", "GreedyBestFirstSearch",
    "astar_search", "AStarSearch",
]
