"""Profile generation tests. Owner: Person B."""
import pandas as pd

from common import config
from data.profiles import SLOT_CHOICES, TARGET_CHOICES, build_profiles, clicks_per_session_rule, synthetic_profiles


def test_synthetic_profiles_reproducible_and_valid():
    a, b = synthetic_profiles(30, seed=1), synthetic_profiles(30, seed=1)
    assert a == b and len(a) == 30
    for p in a:
        assert p.slots_per_week in SLOT_CHOICES and p.target in TARGET_CHOICES
        assert [x.deadline_week for x in p.assignments] == sorted(x.deadline_week for x in p.assignments)


def test_build_profiles_from_feature_table(monkeypatch):
    monkeypatch.setattr(config, "TRAIN_PRESENTATIONS", ["2013J"])
    monkeypatch.setattr(config, "TEST_PRESENTATION", "2014J")
    rows = []
    for pres in ["2013J", "2014J"]:
        for sid in range(40):
            for k, week in enumerate([4, 9, 15]):
                rows.append({"id_student": sid, "code_presentation": pres, "id_assessment": 100 + k,
                             "is_tma": 1, "weight": 20.0, "weeks_since_start": week, "score": 60.0,
                             "prev_score_mean": 55.0, "clicks_4w_content": 120.0,
                             "clicks_4w_interactive": 60.0, "clicks_4w_assessment_prep": 60.0})
    df = pd.DataFrame(rows)
    ps = build_profiles(df, n=30, seed=0)
    assert len(ps) == 30 and len({p.student_id for p in ps}) == 30
    assert [a.deadline_week for a in ps[0].assignments] == [4, 9, 15]
    assert ps[0].clicks_per_session == clicks_per_session_rule(df[df.code_presentation == "2013J"]) == 20
