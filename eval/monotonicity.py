"""Monotonicity check and violation report for the NN (project deliverable P2). Owner: Abdo.

For every profile and assignment, predict the mark with 0, 1, ..., max_sessions sessions
(other assignments at 0). A violation is a step where one more session LOWERS the predicted
mark by more than --tol marks. Also reports how much the mark moves, since a flat NN gives the
search nothing to optimise.

    python -m eval.monotonicity --real --n 30
Writes results/monotonicity_steps.csv and results/monotonicity_summary.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import config
from eval import harness


def check(profiles, predictor, max_sessions: int = config.MAX_SESSIONS_PER_ASSIGNMENT,
          tol: float = 0.0) -> pd.DataFrame:
    rows = []
    for p in profiles:
        n = len(p.assignments)
        for i, a in enumerate(p.assignments):
            marks = []
            for s in range(max_sessions + 1):
                plan = [0] * n
                plan[i] = s
                marks.append(float(predictor.predict(p, tuple(plan))[i]))
            for s in range(max_sessions):
                gain = marks[s + 1] - marks[s]
                rows.append({"student_id": p.student_id, "assignment": a.id, "sessions": s + 1,
                             "mark": marks[s + 1], "gain": gain, "violation": int(gain < -tol)})
    return pd.DataFrame(rows)


def summarize(steps: pd.DataFrame) -> dict:
    per_curve = steps.groupby(["student_id", "assignment"])
    total = per_curve["gain"].sum()
    return {
        "curves": len(total),
        "steps": len(steps),
        "violating_steps_pct": 100 * steps["violation"].mean(),
        "curves_with_violation_pct": 100 * per_curve["violation"].max().mean(),
        "worst_drop": float(steps["gain"].min()),
        "mean_gain_per_session": float(steps["gain"].mean()),
        "max_gain_per_session": float(steps["gain"].max()),
        "mean_gain_0_to_max": float(total.mean()),
        "flat_curves_pct": 100 * float((total.abs() < 1.0).mean()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--tol", type=float, default=0.0)
    args = ap.parse_args(argv)

    predictor = harness.load_predictor("nn")
    steps = check(harness.load_profiles(args.n, args.seed, args.real), predictor, tol=args.tol)
    summary = summarize(steps)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    steps.to_csv(config.RESULTS_DIR / "monotonicity_steps.csv", index=False)
    pd.DataFrame([{"predictor": predictor.name, **summary}]).to_csv(
        config.RESULTS_DIR / "monotonicity_summary.csv", index=False)
    print(f"Monotonicity ({predictor.name}): {summary['curves']} curves, "
          f"{summary['violating_steps_pct']:.1f}% of steps lower the mark, "
          f"{summary['curves_with_violation_pct']:.1f}% of curves have a drop "
          f"(worst {summary['worst_drop']:.2f}).")
    print(f"Mark change per session: mean {summary['mean_gain_per_session']:.2f}, "
          f"max {summary['max_gain_per_session']:.2f}; 0 -> {config.MAX_SESSIONS_PER_ASSIGNMENT} "
          f"sessions: mean +{summary['mean_gain_0_to_max']:.1f}; "
          f"{summary['flat_curves_pct']:.0f}% of curves move < 1 mark.")


if __name__ == "__main__":
    main()
