"""Required planner cases: 5 normal, 3 edge, 2 impossible. Owner: Person B (Week 6).

Remove the skip marks as the real planner lands.
"""
import pytest

pending = pytest.mark.skip(reason="planner not implemented yet (M3, Week 5)")


@pending
@pytest.mark.parametrize("case", ["brief_example", "strong_student", "weak_student", "two_slots", "four_slots"])
def test_normal(case):
    ...


@pending
@pytest.mark.parametrize("case", ["target_already_met", "deadline_first_week", "exactly_enough_slots"])
def test_edge(case):
    ...


@pending
@pytest.mark.parametrize("case", ["target_100_unreachable_by_marks", "too_few_slots_before_deadline"])
def test_impossible(case):
    # must print: "No solution: target unreachable at this capacity" + best predicted marks
    ...
