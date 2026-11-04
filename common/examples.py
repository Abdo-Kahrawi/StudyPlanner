"""Hand-made profiles used by tests and the demo before real profiles exist."""
from common.types import Assignment, StudentProfile


def example_profile(slots_per_week: int = 3, target: float = 55.0) -> StudentProfile:
    """The worked example from the brief: 3 assignments, 3 free slots per week, target 55."""
    return StudentProfile(
        student_id=0,
        history={"prev_score_mean": 60.0},
        assignments=(
            Assignment("A1", "TMA", 20.0, deadline_week=4),
            Assignment("A2", "TMA", 30.0, deadline_week=8),
            Assignment("A3", "TMA", 50.0, deadline_week=12),
        ),
        slots_per_week=slots_per_week,
        start_week=0,
        target=target,
        clicks_per_session=50,
    )
