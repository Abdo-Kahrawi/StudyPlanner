"""Build student profiles for the planner. Owner: Person B.

Real fields come from the processed OULAD tables (data/processed/train_data.csv and
test_data.csv, built by data/make_data.py; columns in data/schema.py). Generated fields follow the rules below, with a fixed seed,
and are documented in docs/data_dictionary.md.

Generated-field rules (GENERATION_RULES):
    slots_per_week     : uniform choice from {2, 3, 4}
    target             : uniform choice from {50, 55, 60, 65, 70}
    clicks_per_session : one third of a typical week of activity in the course =
                         median over the training rows of (sum of the 12 weekly click
                         features) / 4 weeks / 3, rounded, minimum 10
    start_week         : 0 (planning starts at the beginning of the presentation)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import config
from common.types import Assignment, StudentProfile
from data.schema import CLICK_FEATURES

SLOT_CHOICES = (2, 3, 4)
TARGET_CHOICES = (50.0, 55.0, 60.0, 65.0, 70.0)
DEFAULT_CLICKS_PER_SESSION = 50
CLICK_COLS = tuple(CLICK_FEATURES)
ROW_FEATURES = ("prev_score", "cumulative_avg_score", *CLICK_COLS)


def clicks_per_session_rule(train_df: pd.DataFrame | None) -> int:
    if train_df is None or not set(CLICK_COLS) <= set(train_df.columns) or train_df.empty:
        return DEFAULT_CLICKS_PER_SESSION
    weekly = train_df[list(CLICK_COLS)].sum(axis=1) / 4
    return max(10, int(round(float(weekly.median()) / 3)))


def _history(rows: pd.DataFrame) -> dict[str, float]:
    """Per-assignment NN inputs under "<id_assessment>/<feature>" (read by nn/effort.py),
    plus prev_score_mean = the student's average before the first planned assignment."""
    out: dict[str, float] = {}
    for _, r in rows.iterrows():
        for k in ROW_FEATURES:
            if k in r and pd.notna(r[k]):
                out[f"{r['id_assessment']}/{k}"] = float(r[k])
        if "target_score" in r and pd.notna(r["target_score"]):   # evaluation only, never an NN input
            out[f"{r['id_assessment']}/actual_score"] = float(r["target_score"])
    first = rows.iloc[0]
    for k in ("cumulative_avg_score", "prev_score"):
        if k in first and pd.notna(first[k]) and float(first[k]) > 0:
            out["prev_score_mean"] = float(first[k])
            break
    return out


def load_features() -> pd.DataFrame:
    """train_data.csv + test_data.csv from data/make_data.py, as one table."""
    paths = [config.PROCESSED_DIR / "train_data.csv", config.PROCESSED_DIR / "test_data.csv"]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{missing[0]} not found. Run `python -m data.make_data` first, "
                                "or use synthetic_profiles() for development.")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def build_profiles(features: pd.DataFrame | None = None, n: int = 30, seed: int = config.SEED,
                   n_assignments: int = config.N_ASSIGNMENTS) -> list[StudentProfile]:
    """Profiles for n students of the test presentation, each with their first n_assignments TMAs.

    features defaults to data/processed/{train,test}_data.csv. Students are sampled with `seed`.
    """
    if features is None:
        features = load_features()

    rng = np.random.default_rng(seed)
    train = features[features["code_presentation"].isin(config.TRAIN_PRESENTATIONS)]
    cps = clicks_per_session_rule(train)
    test = features[features["code_presentation"] == config.TEST_PRESENTATION]
    if "assessment_type_TMA" in test.columns:
        test = test[test["assessment_type_TMA"].astype(bool)]

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
            student_id=int(sid), history=_history(rows), assignments=assignments,
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
