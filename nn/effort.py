"""
effort.py - Translates hypothetical study sessions from Search into modified NN vector.

Concept from spec:
- Search starts with 0 sessions and adds 1 at a time.
- Each session = extra online activity (clicks).
- We must translate that into the same feature space the NN was trained on:
  4 weeks of clicks split into 3 activity groups (content, quizzes, other).

Assumptions (documented for report):
- 1 session = EFFORT_CLICKS = 20 clicks (tunable).
- Distribution: 50% content, 20% quizzes, 30% other -> mirrors OULAD content-heavy.
- Recency bias: newest week gets most clicks. Allocation per week:
  week4 (closest to deadline) 50%, week3 30%, week2 15%, week1 5%.
- prev_score, cumulative_avg, weight, weeks_since_start, assessment_type unchanged.
- Monotonicity guarantee: clicks only increase, so predicted score should not decrease
  if model is well-behaved (you will test this).

Supports:
- proposed_plan as int (extra sessions for this assessment)
- proposed_plan as dict {id_assessment: sessions} (for multi-assignment search)
- student_profile as dict or pandas Series or np vector
"""

from typing import Union, Dict
import numpy as np
import pandas as pd

EFFORT_CLICKS_PER_SESSION = 20
CLICK_DISTRIB = {"content": 0.5, "quizzes": 0.2, "other": 0.3}
WEEK_WEIGHTS = np.array([0.05, 0.15, 0.30, 0.50])  # week1..week4

# feature order MUST match train_data.csv excluding id_* and target_score
FEATURE_ORDER = [
    "prev_score",
    "cumulative_avg_score",
    "weight",
    "weeks_since_start",
    "group_content_week1", "group_content_week2", "group_content_week3", "group_content_week4",
    "group_quizzes_week1", "group_quizzes_week2", "group_quizzes_week3", "group_quizzes_week4",
    "group_other_week1", "group_other_week2", "group_other_week3", "group_other_week4",
    "assessment_type_CMA", "assessment_type_TMA"
]

CLICK_COLS = [c for c in FEATURE_ORDER if c.startswith("group_")]

class EffortTranslator:
    def __init__(self, clicks_per_session: int = EFFORT_CLICKS_PER_SESSION):
        self.clicks_per_session = clicks_per_session

    def apply_effort(self, base_vector: np.ndarray, extra_sessions: int) -> np.ndarray:
        """Add effort clicks to a feature vector."""
        if extra_sessions <= 0:
            return base_vector.copy()
        total = extra_sessions * self.clicks_per_session
        vec = base_vector.copy().astype(float)

        # indices for click groups
        idx_content = [FEATURE_ORDER.index(f"group_content_week{i}") for i in [1,2,3,4]]
        idx_quiz = [FEATURE_ORDER.index(f"group_quizzes_week{i}") for i in [1,2,3,4]]
        idx_other = [FEATURE_ORDER.index(f"group_other_week{i}") for i in [1,2,3,4]]

        for grp_idx, frac in [(idx_content, CLICK_DISTRIB["content"]),
                              (idx_quiz, CLICK_DISTRIB["quizzes"]),
                              (idx_other, CLICK_DISTRIB["other"])]:
            grp_total = total * frac
            for w, week_frac in enumerate(WEEK_WEIGHTS):
                vec[grp_idx[w]] += grp_total * week_frac
        return vec

def _profile_to_vector(student_profile: Union[Dict, pd.Series, np.ndarray]) -> np.ndarray:
    """Convert dict/Series to ordered np vector."""
    if isinstance(student_profile, np.ndarray):
        return student_profile.astype(float)
    if isinstance(student_profile, dict):
        # dict can be full row or partial
        return np.array([float(student_profile.get(k, 0.0)) for k in FEATURE_ORDER], dtype=float)
    if isinstance(student_profile, pd.Series):
        return np.array([float(student_profile.get(k, 0.0)) for k in FEATURE_ORDER], dtype=float)
    raise TypeError("student_profile must be dict, Series, or np.ndarray")

def build_feature_vector(student_profile: Union[Dict, pd.Series, np.ndarray],
                         proposed_plan: Union[int, Dict] = 0) -> np.ndarray:
    """
    Public API used by predictor.py
    If proposed_plan is int -> apply to this profile.
    If dict -> expects this profile's id_assessment inside, but fallback to 0.
    """
    base = _profile_to_vector(student_profile)
    if isinstance(proposed_plan, dict):
        # If dict like {assessment_id: sessions}, caller should extract. For single vector we default 0.
        extra = 0
        # try to find sessions if profile dict has id_assessment
        if isinstance(student_profile, dict) and "id_assessment" in student_profile:
            extra = proposed_plan.get(student_profile["id_assessment"], 0)
        elif isinstance(student_profile, pd.Series) and "id_assessment" in student_profile:
            extra = proposed_plan.get(student_profile["id_assessment"], 0)
        else:
            extra = proposed_plan.get("default", 0)
        return EffortTranslator().apply_effort(base, extra)
    else:
        return EffortTranslator().apply_effort(base, int(proposed_plan))

def apply_effort(base_vector: np.ndarray, extra_sessions: int) -> np.ndarray:
    """Functional shortcut."""
    return EffortTranslator().apply_effort(base_vector, extra_sessions)


# ---------------------------------------------------------------------------
# Contract bridge: common.types.StudentProfile + Plan -> one NN row per assignment.
# ---------------------------------------------------------------------------
def assignment_base_row(profile, idx: int) -> Dict[str, float]:
    """Feature dict (FEATURE_ORDER keys) for assignment idx of a StudentProfile, before any
    planned sessions.

    profile.history may hold per-assignment values under "<assignment id>/<feature>"
    (data/profiles.py writes these from the real rows) and student-level values under
    "<feature>". Missing previous-score features fall back to "prev_score_mean";
    missing click features are 0.
    """
    a = profile.assignments[idx]
    h = profile.history

    def get(name, default=0.0):
        return float(h.get(f"{a.id}/{name}", h.get(name, default)))

    mean = float(h.get("prev_score_mean", 0.0))
    row = {
        "prev_score": get("prev_score", mean),
        "cumulative_avg_score": get("cumulative_avg_score", mean),
        "weight": float(a.weight),
        "weeks_since_start": float(a.deadline_week),
        "assessment_type_CMA": 1.0 if a.kind == "CMA" else 0.0,
        "assessment_type_TMA": 1.0 if a.kind == "TMA" else 0.0,
    }
    for c in CLICK_COLS:
        row[c] = get(c, 0.0)
    return row


def features_for_plan(profile, plan) -> np.ndarray:
    """NN input matrix [n_assignments, len(FEATURE_ORDER)] for a plan.

    Each planned session for an assignment adds profile.clicks_per_session clicks to that
    assignment's 4-week window, split with CLICK_DISTRIB and WEEK_WEIGHTS above.
    """
    translator = EffortTranslator(clicks_per_session=profile.clicks_per_session)
    rows = [translator.apply_effort(_profile_to_vector(assignment_base_row(profile, i)), int(n))
            for i, n in enumerate(plan)]
    return np.vstack(rows)
