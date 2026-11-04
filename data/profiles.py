"""Build 30+ student profiles from the test presentation. Owner: Person B (Week 3).

Generated fields (document every rule in docs/data_dictionary.md):
    slots_per_week     : drawn from {2, 3, 4}
    target             : e.g. 55 by default, or student's prev mean + 10, capped at 100
    clicks_per_session : e.g. median weekly clicks of the course / 3
"""
from common.types import StudentProfile


def build_profiles(n: int = 30, seed: int = 42) -> list[StudentProfile]:
    raise NotImplementedError("Person B, Week 3")
