"""Run an experiment over seeds; write raw rows and a mean/std table. Owner: Person B.

An experiment is a function `fn(seed) -> list[dict]` returning one dict per measured run.
The harness adds the seed, saves results/<name>_raw.csv and results/<name>_summary.csv.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from common import config


def run(name: str, fn: Callable[[int], list[dict]], seeds=config.EXPERIMENT_SEEDS,
        verbose: bool = True) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        if verbose:
            print(f"[{name}] seed {seed}")
        for r in fn(seed):
            rows.append({"seed": seed, **r})
    df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / f"{name}_raw.csv", index=False)
    return df


def summarize(df: pd.DataFrame, by: list[str], metrics: list[str], name: str | None = None) -> pd.DataFrame:
    """Mean and standard deviation of each metric over seeds.

    First averages within each seed (so every seed counts once), then takes mean and std
    across the seeds.
    """
    per_seed = df.groupby(["seed", *by], as_index=False)[metrics].mean()
    out = per_seed.groupby(by)[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out = out.reset_index()
    if name:
        out.to_csv(config.RESULTS_DIR / f"{name}_summary.csv", index=False)
    return out


def load_predictor(kind: str = "nn"):
    """The requested predictor, or the stub if Person A's model is not ready yet."""
    from nn.predictor import AverageMarkPredictor, MarkPredictor
    from common.stubs import HistoryFakePredictor
    cls = MarkPredictor if kind == "nn" else AverageMarkPredictor
    pred = cls()
    try:
        from common.examples import example_profile
        pred.predict(example_profile(), (0, 0, 0))
        return pred
    except (NotImplementedError, FileNotFoundError, OSError):
        print(f"note: '{kind}' predictor not ready, using the stub HistoryFakePredictor")
        return HistoryFakePredictor()


def load_search():
    """Person A's search, or the brute-force stub if it is not ready yet."""
    from search import search
    from common.examples import example_profile
    from common.stubs import FakePredictor, reference_search
    try:
        search(example_profile(), FakePredictor(), "ucs", lambda p: True, 3)
        return search
    except NotImplementedError:
        print("note: search not ready, using the stub reference_search")
        return reference_search


def load_profiles(n: int, seed: int, real: bool):
    from data.profiles import build_profiles, synthetic_profiles
    return build_profiles(n=n, seed=seed) if real else synthetic_profiles(n=n, seed=seed)
