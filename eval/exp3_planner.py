"""Experiment 3 inside the planner: NN vs average-mark predictor on the real OULAD profiles.

Follows the project plan: UCS, CSP = BT+FC+AC-3, 3 slots per week, same profiles and
seeds; only the predictor changes. Targets 60 / 70 / 80 as in exp1_planner. Predictors:
    nn       the monotone NN used by the planner (nn/weights/)
    nn-mlp   the plain MLP (nn/weights/mlp/), if trained: flat in effort
    avg      the student's previous average, ignores effort
Measures % of profiles solved, plan cost, and how often the predictor's "no sessions
needed" call (all predicted marks >= target with 0 sessions) agrees with the student's
ACTUAL held-out marks (all actual marks >= target). Standalone MAE / RMSE is in
eval/exp3_nn_vs_avg.py (Person A).

    python -m eval.exp3_planner --real
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from common import config
from eval import harness, plots
from eval.exp1_planner import TARGETS
from integration.planner import plan_for

METRICS = ["solved", "cost", "call_correct", "nodes_expanded"]


def load_predictors() -> dict:
    preds = {"nn": harness.load_predictor("nn"), "avg": harness.load_predictor("avg")}
    mlp = config.WEIGHTS_DIR / "mlp"
    try:
        from nn.predictor import MarkPredictor
        p = MarkPredictor(str(mlp / "nn_weights.pt"), str(mlp / "scaler.npz"))
        p.name = "nn-mlp"
        preds["nn-mlp"] = p
    except (ImportError, FileNotFoundError):
        print("note: no plain-MLP weights in nn/weights/mlp/, skipping nn-mlp")
    return preds


def actual_marks(profile) -> np.ndarray | None:
    vals = [profile.history.get(f"{a.id}/actual_score") for a in profile.assignments]
    return None if any(v is None for v in vals) else np.array(vals, dtype=float)


def make_experiment(real: bool, n_profiles: int, predictors: dict, search_fn):
    def experiment(seed: int) -> list[dict]:
        rows = []
        for base in harness.load_profiles(n_profiles, seed, real):
            actual = actual_marks(base)
            for target in TARGETS:
                prof = dataclasses.replace(base, target=target, slots_per_week=3)
                for name, pred in predictors.items():
                    out = plan_for(prof, pred, "ucs", search_fn=search_fn)
                    zero_ok = bool(np.all(pred.predict(prof, (0,) * len(prof.assignments)) >= target))
                    rows.append({
                        "profile": base.student_id, "target": target, "predictor": name,
                        "solved": int(out.found), "cost": sum(out.plan) if out.found else np.nan,
                        "call_correct": (np.nan if actual is None
                                         else int(zero_ok == bool(np.all(actual >= target)))),
                        "nodes_expanded": out.search.nodes_expanded})
        return rows
    return experiment


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--profiles", type=int, default=30)
    args = ap.parse_args(argv)
    predictors = load_predictors()
    df = harness.run("exp3_planner", make_experiment(args.real, args.profiles, predictors,
                                                     harness.load_search()))
    summary = harness.summarize(df, ["target", "predictor"], METRICS, name="exp3_planner")
    print(summary.round(2).to_string(index=False))
    for metric, ylabel in [("solved", "share of profiles with a plan"),
                           ("cost", "plan cost: total sessions (solved only)"),
                           ("call_correct", "'0 sessions enough?' matches actual marks")]:
        plots.grouped_bars(summary, "target", "predictor", metric,
                           config.RESULTS_DIR / f"exp3_planner_{metric}.png",
                           f"Experiment 3: {metric.replace('_', ' ')} by predictor", ylabel)
    print(f"saved tables and plots to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
