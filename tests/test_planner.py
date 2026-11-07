"""Required planner cases: 5 normal, 3 edge, 2 impossible. Owner: Person B.

Uses the stub predictor and stub search until Person A's modules land; switch the two
fixtures below to the real ones at integration time.
"""
import dataclasses

import numpy as np
import pytest

from common.examples import example_profile
from common.stubs import FakePredictor, HistoryFakePredictor, reference_search
from common.types import Assignment
from integration.planner import NO_SOLUTION, plan_for


@pytest.fixture
def search_fn():
    return reference_search


def run(profile, search_fn, predictor=None):
    return plan_for(profile, predictor or FakePredictor(), "ucs", search_fn=search_fn)


def assert_valid_solution(out):
    p = out.profile
    assert out.found and out.schedule.ok
    assert np.all(out.predicted >= p.target)
    assert len(out.schedule.timetable) == sum(out.plan)
    deadlines = {a.id: a.deadline_week for a in p.assignments}
    assert all(w < deadlines[aid] for (aid, _), (w, _) in out.schedule.timetable.items())


def with_history(mean, **kw):
    return dataclasses.replace(example_profile(**kw), history={"prev_score_mean": mean})


# ---- 5 normal cases -------------------------------------------------------------
def test_normal_brief_example(search_fn):
    out = run(example_profile(), search_fn)
    assert_valid_solution(out)
    assert out.plan == (3, 3, 3)                       # FakePredictor: 3 sessions reach 55.4


def test_normal_strong_student_needs_fewer(search_fn):
    strong = run(with_history(80), search_fn, HistoryFakePredictor())
    weak = run(with_history(60), search_fn, HistoryFakePredictor())
    assert_valid_solution(strong) and assert_valid_solution(weak)
    assert sum(strong.plan) < sum(weak.plan)


def test_normal_weak_student(search_fn):
    out = run(with_history(55, target=50.0), search_fn, HistoryFakePredictor())
    assert_valid_solution(out)


def test_normal_two_slots_per_week(search_fn):
    assert_valid_solution(run(example_profile(slots_per_week=2), search_fn))


def test_normal_four_slots_per_week(search_fn):
    out = run(example_profile(slots_per_week=4), search_fn)
    assert_valid_solution(out)
    assert all(s < 4 for _, s in out.schedule.timetable.values())


# ---- 3 edge cases ---------------------------------------------------------------
def test_edge_target_already_met(search_fn):
    out = run(example_profile(target=40.0), search_fn)       # base mark is 40
    assert_valid_solution(out)
    assert out.plan == (0, 0, 0) and out.schedule.timetable == {}


def test_edge_deadline_in_first_week(search_fn):
    p = dataclasses.replace(example_profile(), assignments=(
        Assignment("A1", "TMA", 20.0, 1), *example_profile().assignments[1:]))
    out = run(p, search_fn)                                  # 3 slots before A1, needs 3
    assert_valid_solution(out)
    assert all(w == 0 for (aid, _), (w, _) in out.schedule.timetable.items() if aid == "A1")


def test_edge_exactly_enough_slots(search_fn):
    p = dataclasses.replace(example_profile(slots_per_week=1), assignments=(
        Assignment("A1", "TMA", 20.0, 3), Assignment("A2", "TMA", 30.0, 6), Assignment("A3", "TMA", 50.0, 9)))
    out = run(p, search_fn)                                  # 9 slots, plan (3, 3, 3)
    assert_valid_solution(out)
    assert len(out.schedule.timetable) == 9


# ---- 2 impossible cases ---------------------------------------------------------
def test_impossible_target_100(search_fn, capsys):
    out = run(example_profile(target=100.0), search_fn)
    assert not out.found
    assert out.message.startswith(NO_SOLUTION)
    assert "Best predicted marks" in out.message
    assert out.predicted is not None and np.all(out.predicted < 100)


def test_impossible_too_few_slots_before_deadline(search_fn):
    p = dataclasses.replace(example_profile(slots_per_week=1), assignments=(
        Assignment("A1", "TMA", 20.0, 2), *example_profile().assignments[1:]))
    out = run(p, search_fn)                                  # A1 needs 3, only 2 slots
    assert not out.found and out.message.startswith(NO_SOLUTION)
    assert out.nn_yes_csp_no                                 # NN said enough, CSP said no
    assert out.schedule is not None and out.schedule.ok      # best plan is still schedulable
