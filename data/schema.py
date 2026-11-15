"""
schema.py - Single source of truth for StudyPlanner (P1) data schema.
Generated from the ACTUAL train_data.csv / test_data.csv (35,601 + 7,399 rows, 22 cols),
NOT from the draft template.

Use everywhere instead of hard-coding column names:
    from schema import FEATURES, TARGET, KEYS, build_feature_row, validate_dataframe
    from schema import ACTIVITY_GROUPS, WEEKS, CLICK_FEATURES, CLICK_4W

Why this file exists:
  The draft template used names like clicks_4w_content / prev_score_mean / is_tma / score.
  None of those exist in the CSVs. This file maps template -> real and makes the
  mismatch explicit so nn/ and search/ never silently use a wrong column.
"""
from __future__ import annotations
from typing import Dict, List

# ---------------------------------------------------------------------------
# Keys / target - these DO match the template
# ---------------------------------------------------------------------------
KEYS = ["id_student", "code_presentation", "id_assessment"]
TARGET = "target_score"  # template said "score" - real column is "target_score", 0-100

# ---------------------------------------------------------------------------
# Activity groups - template vs real
# ---------------------------------------------------------------------------
# Template: ["content", "interactive", "assessment_prep"]
# Real:     ["content", "quizzes", "other"]
# Mapping (best-effort, from OULAD activity types):
#   content         -> content        (homepage / oucontent / resource / url / page / subpage)
#   interactive     -> quizzes        (quiz / questionnaire in OULAD grouping used for make_data.py)
#   assessment_prep -> other          (forumng / ouelluminate / oucollaborate / glossary / ... - everything else)
#   Do NOT rename these in make_data.py without updating this file + nn/effort.py.
ACTIVITY_GROUPS = ["content", "quizzes", "other"]

# Template had only a 4-week aggregate per group (clicks_4w_*).
# Real data is per-week: 4 weeks x 3 groups = 12 click features.
# week1 = 4 weeks before deadline, week4 = week of deadline (most recent).
WEEKS = [1, 2, 3, 4]

# ---------------------------------------------------------------------------
# FEATURES - the 18 columns the NN actually sees, in EXACT order used for training.
# Order matters: nn/model.py INPUT_DIM=18, nn/effort.py FEATURE_ORDER must match this.
# ---------------------------------------------------------------------------
FEATURES = [
    # previous performance (2) - template had 3: prev_score_mean / prev_score_last / n_prev_assessments
    # Real data collapses those to:
    "prev_score",            # score on immediately previous assessment, 0 if none yet
    "cumulative_avg_score",  # mean of all previous assessments, 0 if none yet
    # assessment descriptors (2) - template had is_tma only
    "weight",                # assessment weight (1, 5, 10, 18, 20, 35, ...)
    "weeks_since_start",     # week of course when assessment is due
    # clicks in 4 weeks before deadline, per activity group (12)
    "group_content_week1", "group_content_week2", "group_content_week3", "group_content_week4",
    "group_quizzes_week1", "group_quizzes_week2", "group_quizzes_week3", "group_quizzes_week4",
    "group_other_week1",   "group_other_week2",   "group_other_week3",   "group_other_week4",
    # assessment type, one-hot (2) - template had is_tma (single bool)
    "assessment_type_CMA",   # Computer-Marked Assessment
    "assessment_type_TMA",   # Tutor-Marked Assessment
]
assert len(FEATURES) == 18, f"Expected 18 features, got {len(FEATURES)}"

# Convenience slices
PREV_PERFORMANCE = ["prev_score", "cumulative_avg_score"]
ASSESSMENT_DESCRIPTORS = ["weight", "weeks_since_start"]
ASSESSMENT_TYPE = ["assessment_type_CMA", "assessment_type_TMA"]

CLICK_FEATURES = [c for c in FEATURES if c.startswith("group_")]
CONTENT_FEATURES = [c for c in CLICK_FEATURES if "content" in c]
QUIZZES_FEATURES = [c for c in CLICK_FEATURES if "quizzes" in c]
OTHER_FEATURES = [c for c in CLICK_FEATURES if "other" in c]

# 4-week totals per group, derived (not in CSV, useful for reporting / sanity checks)
CLICK_4W = {
    "content": CONTENT_FEATURES,
    "quizzes": QUIZZES_FEATURES,
    "other": OTHER_FEATURES,
}

ALL_COLUMNS = KEYS + [TARGET] + FEATURES
# KEYS (3) + TARGET (1) + FEATURES (18) = 22 columns, matches CSVs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def click_feature(group: str, week: int) -> str:
    """e.g. click_feature('content', 4) -> 'group_content_week4'"""
    assert group in ACTIVITY_GROUPS, f"group must be one of {ACTIVITY_GROUPS}"
    assert week in WEEKS, f"week must be one of {WEEKS}"
    return f"group_{group}_week{week}"

def build_feature_row(row: Dict) -> Dict:
    """Extract the 18 NN features from a full row dict / DataFrame row (as dict)."""
    return {k: row.get(k, 0) for k in FEATURES}

def feature_vector(row: Dict) -> List[float]:
    """Ordered list[float] of length 18, in FEATURES order. For nn/effort.py."""
    return [float(row.get(k, 0) or 0) for k in FEATURES]

def validate_dataframe(df, name: str = "df") -> None:
    """Raise a clear error if a DataFrame is missing expected columns."""
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns {missing}. Expected {ALL_COLUMNS}, got {list(df.columns)}")
    # Type sanity: assessment_type_* should be bool
    for c in ASSESSMENT_TYPE:
        if df[c].dtype != bool:
            # pandas may read them as bool already; if not, coerce in make_data.py
            pass

# ---------------------------------------------------------------------------
# Data notes (from actual CSV inspection, for report)
# ---------------------------------------------------------------------------
DATA_NOTES = """
- train_data.csv: 35,601 rows, code_presentation 2013B only in sample, id_assessment 14984-15019.
- test_data.csv: 7,399 rows, code_presentation 2014J, id_assessment 15020-15024.
- WARNING: in test_data.csv all 12 group_* click columns are 0.0. If that is real,
  the NN cannot use engagement on test; if it is a make_data.py bug, fix it before
  reporting test RMSE. Train group_other_* has heavy outliers (max 2126 clicks).
- prev_score / cumulative_avg_score are 0.0 for first assessment (no history) - not missing.
- target_score mean: train ~78.9, test ~65.7 - distribution shift, note in report.
"""
