"""Experiment 1 inside the planner: UCS vs Greedy vs A* on the real OULAD profiles.

Follows the project plan: same 30 profiles per seed (5 seeds), NN predictor, CSP =
BT+FC+AC-3, 3 slots per week. Most real students already reach the generated targets
(50-70) with 0 sessions, so the target is set to three difficulty levels (60, 70, 80) to
give search real work. Measures nodes expanded / generated, plan cost (total sessions),
% of profiles solved and runtime; cost excess is measured against UCS (optimal).

eval/exp1_search.py is Person A's standalone version (training rows, no CSP).

    python -m eval.exp1_planner --real
Writes results/exp1_planner_raw.csv, results/exp1_planner_summary.csv and plots.
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from common import config
from eval import harness, plots
from integration.planner import plan_for

STRATEGIES = {"ucs": "UCS", "greedy": "Greedy", "astar": "A*"}
TARGETS = (60.0, 70.0, 80.0)
METRICS = ["solved", "cost", "excess_over_ucs", "nodes_expanded", "nodes_solved", "nodes_unsolved",
           "nodes_generated", "seconds"]


def make_experiment(real: bool, n_profiles: int, predictor, search_fn):
    def experiment(seed: int) -> list[dict]:
        rows = []
        for base in harness.load_profiles(n_profiles, seed, real):
            for target in TARGETS:
                prof = dataclasses.replace(base, target=target, slots_per_week=3)
                ucs_cost = None
                for strategy in STRATEGIES:
                    out = plan_for(prof, predictor, strategy, search_fn=search_fn)
                    cost = sum(out.plan) if out.found else np.nan
                    if strategy == "ucs":
                        ucs_cost = cost
                    rows.append({"profile": base.student_id, "target": target, "strategy": strategy,
                                 "solved": int(out.found), "cost": cost,
                                 "excess_over_ucs": cost - ucs_cost,
                                 "nodes_expanded": out.search.nodes_expanded,
                                 # split: an unsolved profile always explores the whole capped space
                                 "nodes_solved": out.search.nodes_expanded if out.found else np.nan,
                                 "nodes_unsolved": np.nan if out.found else out.search.nodes_expanded,
                                 "nodes_generated": out.search.nodes_generated,
                                 "seconds": out.seconds})
        return rows
    return experiment


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--profiles", type=int, default=30)
    args = ap.parse_args(argv)
    predictor, search_fn = harness.load_predictor("nn"), harness.load_search()
    df = harness.run("exp1_planner", make_experiment(args.real, args.profiles, predictor, search_fn))
    summary = harness.summarize(df, ["target", "strategy"], METRICS, name="exp1_planner")
    order = {s: i for i, s in enumerate(STRATEGIES)}
    summary = summary.sort_values(["target", "strategy"],
                                  key=lambda c: c.map(order) if c.name == "strategy" else c)
    print(summary.round(2).to_string(index=False))
    for metric, ylabel, log in [("nodes_solved", "nodes expanded, solved profiles", True),
                                ("cost", "plan cost: total sessions (solved only)", False)]:
        plots.grouped_bars(summary, "target", "strategy", metric,
                           config.RESULTS_DIR / f"exp1_planner_{metric}.png",
                           f"Experiment 1: {metric.replace('_', ' ')} by search strategy", ylabel,
                           log=log, series_labels=STRATEGIES)
    print(f"saved tables and plots to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
