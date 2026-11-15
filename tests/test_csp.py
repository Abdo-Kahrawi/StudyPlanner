"""CSP unit tests: BT, FC, AC-3 switches and counters. Owner: Person B."""
import dataclasses

import pytest

from common.examples import example_profile
from common.types import Assignment
from csp.ac3 import ac3
from csp.solver import MODES, capacity_ok, schedule
from csp.variables import build_csp


def tiny_profile():
    """2 slots/week, A1 due end of week 1, A2 due end of week 2: 2 + 2 slots."""
    return dataclasses.replace(example_profile(slots_per_week=2), assignments=(
        Assignment("A1", "TMA", 50.0, 1), Assignment("A2", "TMA", 50.0, 2)))


def check_valid(res, profile, plan):
    deadlines = {a.id: a.deadline_week for a in profile.assignments}
    assert len(res.timetable) == sum(plan)
    assert len(set(res.timetable.values())) == len(res.timetable)       # one session per slot
    for (aid, k), (week, slot) in res.timetable.items():
        assert profile.start_week <= week < deadlines[aid]              # before its deadline
        assert 0 <= slot < profile.slots_per_week
        if k > 0:                                                       # in order
            assert res.timetable[(aid, k - 1)] < (week, slot)


@pytest.mark.parametrize("mode", MODES)
def test_known_answer_tiny(mode):
    """Plan (2, 2) fits exactly: A1 must take both week-1 slots, A2 both week-2 slots."""
    fc, ac = MODES[mode]
    p = tiny_profile()
    res = schedule((2, 2), p, use_fc=fc, use_ac3=ac)
    assert res.ok
    assert res.timetable == {("A1", 0): (0, 0), ("A1", 1): (0, 1), ("A2", 0): (1, 0), ("A2", 1): (1, 1)}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("plan", [(4, 5, 3), (1, 0, 0), (12, 12, 12), (0, 0, 7)])
def test_valid_schedules(mode, plan):
    fc, ac = MODES[mode]
    p = example_profile()
    res = schedule(plan, p, use_fc=fc, use_ac3=ac)
    assert res.ok
    check_valid(res, p, plan)


@pytest.mark.parametrize("mode", MODES)
def test_infeasible_gives_no_solution(mode):
    fc, ac = MODES[mode]
    res = schedule((3, 2), tiny_profile(), use_fc=fc, use_ac3=ac)
    assert not res.ok and res.timetable == {}


def test_counters_show_pruning_differences():
    p = example_profile()
    bt = schedule((13, 0, 0), p, use_fc=False, use_ac3=False)
    fc = schedule((13, 0, 0), p, use_fc=True, use_ac3=False)
    ac = schedule((13, 0, 0), p, use_fc=True, use_ac3=True)
    assert not (bt.ok or fc.ok or ac.ok)
    assert bt.backtracks > 0 and bt.values_pruned == 0
    assert fc.values_pruned > 0
    assert ac.assignments == 0 < bt.assignments                 # AC-3 proves it before search


def test_ac3_trims_order_bounds():
    """With 3 ordered sessions in 6 slots, session 0 can't take the last 2 slots."""
    p = dataclasses.replace(example_profile(slots_per_week=3), assignments=(
        Assignment("A1", "TMA", 100.0, 2),))
    csp = build_csp((3,), p)
    doms = {v: list(d) for v, d in csp.domains.items()}
    stats = {"values_pruned": 0, "ac3_removed": 0, "revisions": 0}
    assert ac3(csp, doms, stats)
    assert doms[("A1", 0)] == [0, 1, 2, 3]
    assert doms[("A1", 2)] == [2, 3, 4, 5]


def test_mac_extension_matches():
    res = schedule((4, 5, 3), example_profile(), use_fc=True, use_ac3=True, use_mac=True)
    assert res.ok
    check_valid(res, example_profile(), (4, 5, 3))


def test_capacity_check_and_limit():
    p = example_profile()
    assert capacity_ok((12, 12, 12), p) and not capacity_ok((13, 0, 0), p)
    res = schedule((13, 0, 0), p, use_fc=False, use_ac3=False, max_assignments=50)
    assert not res.ok and any("stopped" in m for m in res.log)
