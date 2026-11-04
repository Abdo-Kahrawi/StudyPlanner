"""Feature names shared by data/ and nn/. Owner: Person A. Frozen at end of Week 1.

Confirm exact OULAD table and column names in Week 1, then fill these in.
Do NOT add gender, age_band, region, disability or imd_band.
"""

# 3 activity groups built from vle.activity_type (mapping fixed in make_data.py)
ACTIVITY_GROUPS = ["content", "interactive", "assessment_prep"]

FEATURES = [
    # clicks in the 4 weeks before the deadline, per activity group
    "clicks_4w_content",
    "clicks_4w_interactive",
    "clicks_4w_assessment_prep",
    # previous performance
    "prev_score_mean",
    "prev_score_last",
    "n_prev_assessments",
    # assessment descriptors
    "is_tma",
    "weight",
    "weeks_since_start",
    # baseline engagement (whole course so far)
    "clicks_total_so_far",
    "active_days_4w",
    "days_submitted_early_prev",
]
TARGET = "score"
KEYS = ["id_student", "code_presentation", "id_assessment"]
