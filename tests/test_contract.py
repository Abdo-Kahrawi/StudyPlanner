"""Checks the shared contract and the stubs both people build against."""
import numpy as np

from common.examples import example_profile
from common.stubs import FakePredictor, capacity_schedule


def test_fake_predictor_shape_and_range():
    p = example_profile()
    marks = FakePredictor().predict(p, (0, 0, 0))
    assert marks.shape == (3,)
    assert np.all((marks >= 0) & (marks <= 100))


def test_fake_predictor_is_monotone():
    p, f = example_profile(), FakePredictor()
    for n in range(10):
        assert np.all(f.predict(p, (n + 1,) * 3) >= f.predict(p, (n,) * 3))


def test_capacity_schedule_respects_deadlines():
    p = example_profile()
    res = capacity_schedule((4, 5, 3), p)
    assert res.ok
    deadlines = {a.id: a.deadline_week for a in p.assignments}
    assert all(week < deadlines[aid] for (aid, _), (week, _) in res.timetable.items())
    assert len(set(res.timetable.values())) == len(res.timetable)  # one session per slot


def test_capacity_schedule_rejects_overfull_first_deadline():
    p = example_profile(slots_per_week=2)   # only 8 slots before week 4
    assert not capacity_schedule((9, 0, 0), p).ok
