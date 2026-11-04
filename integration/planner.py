"""Search proposes, NN checks marks, CSP checks the schedule. Owner: Person B (Week 4).

Loop:
    1. search() pops a plan.
    2. predictor.predict(plan) >= target for every assignment?
    3. schedule(plan) ok?  (cache results per plan)
    4. Both yes -> done. NN yes but CSP no -> record an "NN yes, CSP no" event, continue.
    5. Nothing left -> "No solution: target unreachable at this capacity" + best predicted marks.
"""
from dataclasses import dataclass, field

from common.types import Predictor, ScheduleResult, SearchResult, StudentProfile


@dataclass
class PlannerOutput:
    profile: StudentProfile
    search: SearchResult
    schedule: ScheduleResult | None
    nn_yes_csp_no: list[tuple[int, ...]] = field(default_factory=list)
    message: str = ""


def plan_for(profile: StudentProfile, predictor: Predictor, strategy: str = "ucs",
             use_fc: bool = True, use_ac3: bool = True) -> PlannerOutput:
    raise NotImplementedError("Person B, Week 4")
