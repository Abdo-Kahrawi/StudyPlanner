"""The 30-profile deliverable runs end to end and every schedule is valid. Owner: Person B."""
from common.stubs import HistoryFakePredictor, reference_search
from data.profiles import synthetic_profiles
from eval.run_profiles import run_profiles


def test_thirty_profiles_produce_outputs():
    profiles = synthetic_profiles(30, seed=0)
    summary, sessions = run_profiles(profiles, HistoryFakePredictor(), reference_search)
    assert len(summary) == 30
    assert set(summary["status"]) <= {"found", "unreachable"}
    found = summary[summary.status == "found"]
    assert (found[["a1_predicted", "a2_predicted", "a3_predicted"]].min(axis=1) >= found["target"]).all()
    # every scheduled session is before its deadline and slots are not shared
    deadlines = {(r.student_id, r[f"a{i}_id"]): r[f"a{i}_deadline_week"]
                 for _, r in summary.iterrows() for i in (1, 2, 3)}
    assert all(s.week <= deadlines[(s.student_id, s.assignment)] for s in sessions.itertuples())
    assert not sessions.duplicated(["student_id", "week", "slot"]).any()
