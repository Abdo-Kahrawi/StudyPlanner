"""Search unit tests incl. a small example with a known answer. Owner: Abdo."""
import numpy as np
import pytest

from common.examples import example_profile
from common.stubs import FakePredictor, capacity_schedule, reference_search
from search import StudyPlannerProblem, astar_search, greedy_search, search, uniform_cost_search


def test_ucs_known_answer():
    # Fallback predictor: base 40 + 2 marks per session -> 8 sessions to reach 55 on each.
    problem = StudyPlannerProblem([{"prev_score": 40.0}] * 2, targets=[55, 55])
    res = uniform_cost_search(problem)
    assert res.success and res.solution_state == (8, 8) and res.cost == 16


def test_astar_matches_ucs_cost():
    mk = lambda: StudyPlannerProblem([{"prev_score": 40.0}, {"prev_score": 50.0}], targets=[55, 55],
                                     max_gain_per_session=2.0)
    ucs, astar = uniform_cost_search(mk()), astar_search(mk())
    assert astar.cost == ucs.cost == 11
    assert astar.nodes_expanded < ucs.nodes_expanded


@pytest.mark.parametrize("strategy", ["ucs", "greedy", "astar"])
def test_contract_search_brief_example(strategy):
    p = example_profile()
    ok = lambda plan: capacity_schedule(plan, p).ok
    res = search(p, FakePredictor(), strategy, ok, 10)
    assert res.status == "found" and np.all(res.predicted >= p.target) and ok(res.plan)
    assert res.trace and res.nodes_expanded == len(res.trace)
    if strategy != "greedy":   # UCS and A* (admissible h) are optimal
        assert res.cost == reference_search(p, FakePredictor(), "ucs", ok, 10).cost


def test_contract_search_unreachable_reports_best_plan():
    p = example_profile(slots_per_week=1, target=95)
    ok = lambda plan: capacity_schedule(plan, p).ok
    res = search(p, FakePredictor(), "astar", ok, 10)
    assert res.status == "unreachable" and res.plan is None
    assert res.best_plan is not None and ok(res.best_plan)


def test_contract_search_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        search(example_profile(), FakePredictor(), "dfs")
