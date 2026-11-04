"""Turn study sessions into extra clicks for the NN inputs. Owner: Person A (Week 5).

Rule: each session adds profile.clicks_per_session clicks to the 4-week window before
that assignment's deadline, split across the 3 activity groups in the student's own
historical ratio. Write this rule into docs/data_dictionary.md.
"""
from common.types import Plan, StudentProfile


def features_for_plan(profile: StudentProfile, plan: Plan):
    raise NotImplementedError("Person A, Week 5")
