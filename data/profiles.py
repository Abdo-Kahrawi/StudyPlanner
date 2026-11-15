"""Build student profiles for the planner. Owner: Person B.

Real fields come from the processed OULAD feature table (data/processed/features.parquet,
built by data/make_data.py). Generated fields follow the rules below, with a fixed seed,
and are documented in docs/data_dictionary.md.

Generated-field rules (GENERATION_RULES):
    slots_per_week     : uniform choice from {2, 3, 4}
    target             : uniform choice from {50, 55, 60, 65, 70}
    clicks_per_session : one third of a typical week of activity in the course =
                         median over the training rows of (sum of the three 4-week click
                         features) / 4 weeks / 3, rounded, minimum 10
    start_week         : 0 (planning starts at the beginning of the presentation)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import config
from common.types import Assignment, StudentProfile

SLOT_CHOICES = (2, 3, 4)
TARGET_CHOICES = (50.0, 55.0, 60.0, 65.0, 70.0)
DEFAULT_CLICKS_PER_SESSION = 50
CLICK_COLS = ("clicks_4w_content", "clicks_4w_interactive", "clicks_4w_assessment_prep")


def clicks_per_session_rule(train_df: pd.DataFrame | None) -> int:
    if train_df is None or not set(CLICK_COLS) <= set(train_df.columns) or train_df.empty:
        return DEFAULT_CLICKS_PER_SESSION
    weekly = train_df[list(CLICK_COLS)].sum(axis=1) / 4
    return max(10, int(round(float(weekly.median()) / 3)))


def _history(first_row: pd.Series) -> dict[str, float]:
    keys = ["prev_score_mean", "prev_score_last", "n_prev_assessments", "clicks_total_so_far",
            "active_days_4w", "days_submitted_early_prev", *CLICK_COLS]
    out = {}
    for k in keys:
        if k in first_row and pd.notna(first_row[k]):
            out[k] = float(first_row[k])
    return out


def build_profiles(features: pd.DataFrame | None = None, n: int = 30, seed: int = config.SEED,
                   n_assignments: int = config.N_ASSIGNMENTS) -> list[StudentProfile]:
    """Profiles for n students of the test presentation, each with their first n_assignments TMAs.

    features defaults to data/processed/features.parquet. Students are sampled with `seed`.
    """
    if features is None:
        path = config.PROCESSED_DIR / "features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m data.make_data` first, "
                                    "or use synthetic_profiles() for development.")
        features = pd.read_parquet(path)

    rng = np.random.default_rng(seed)
    train = features[features["code_presentation"].isin(config.TRAIN_PRESENTATIONS)]
    cps = clicks_per_session_rule(train)
    test = features[features["code_presentation"] == config.TEST_PRESENTATION]
    if "is_tma" in test.columns:
        test = test[test["is_tma"] == 1]

    eligible = [sid for sid, g in test.groupby("id_student") if len(g) >= n_assignments]
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)} students have {n_assignments}+ TMAs; asked for {n}")
    chosen = rng.choice(sorted(eligible), size=n, replace=False)

    profiles = []
    for sid in chosen:
        rows = test[test["id_student"] == sid].sort_values("weeks_since_start").head(n_assignments)
        assignments = tuple(
            Assignment(id=str(r["id_assessment"]), kind="TMA", weight=float(r["weight"]),
                       deadline_week=max(1, int(math.ceil(r["weeks_since_start"]))))
            for _, r in rows.iterrows())
        profiles.append(StudentProfile(
            student_id=int(sid), history=_history(rows.iloc[0]), assignments=assignments,
            slots_per_week=int(rng.choice(SLOT_CHOICES)), start_week=0,
            target=float(rng.choice(TARGET_CHOICES)), clicks_per_session=cps))
    return profiles


def synthetic_profiles(n: int = 30, seed: int = config.SEED) -> list[StudentProfile]:
    """Made-up profiles with realistic shapes, for development and tests before real data exists.

    Deadlines: first TMA in weeks 3-6, then gaps of 3-6 weeks. Previous mean score 35-85.
    These are NOT student records and must not be reported as results.
    """
    rng = np.random.default_rng(seed)
    profiles = []
    for i in range(n):
        d1 = int(rng.integers(3, 7))
        d2 = d1 + int(rng.integers(3, 7))
        d3 = d2 + int(rng.integers(3, 7))
        weights = (20.0, 30.0, 50.0)
        profiles.append(StudentProfile(
            student_id=100_000 + i,
            history={"prev_score_mean": float(rng.uniform(35, 85))},
            assignments=tuple(Assignment(f"T{k + 1}", "TMA", w, d)
                              for k, (w, d) in enumerate(zip(weights, (d1, d2, d3)))),
            slots_per_week=int(rng.choice(SLOT_CHOICES)), start_week=0,
            target=float(rng.choice(TARGET_CHOICES)), clicks_per_session=DEFAULT_CLICKS_PER_SESSION))
    return profiles
