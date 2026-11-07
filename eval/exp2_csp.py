"""Experiment 2: BT vs BT+FC vs BT+FC+AC-3 at 2, 3, 4 slots per week. Owner: Person B.

For each seed: build 30 profiles, let the planner pick a plan at 3 slots per week, then
schedule that same plan with each CSP mode at 2, 3 and 4 slots per week. Fewer slots make
the same plan tighter, so backtracking and pruning show up. The counting pre-check is OFF
here so every mode does its own work.

Usage:
    python -m eval.exp2_csp            # synthetic profiles until real data exists
    python -m eval.exp2_csp --real     # profiles from data/processed/features.parquet
"""
from __future__ import annotations

import argparse
import dataclasses
import time

from common import config
from csp.solver import MODES, schedule
from eval import harness, plots
from integration.planner import plan_for

SLOTS = (2, 3, 4)
LIMIT = 100_000
METRICS = ["assignments", "backtracks", "values_pruned", "seconds", "ok", "hit_limit"]
LABELS = {"bt": "BT", "fc": "BT+FC", "ac3": "BT+FC+AC-3"}


def make_experiment(real: bool, n_profiles: int):
    predictor = harness.load_predictor("nn")
    search_fn = harness.load_search()

    def experiment(seed: int) -> list[dict]:
        rows = []
        for base in harness.load_profiles(n_profiles, seed, real):
            base3 = dataclasses.replace(base, slots_per_week=3)
            plan = plan_for(base3, predictor, "ucs", search_fn=search_fn).plan
            if plan is None or sum(plan) == 0:
                continue
            for slots in SLOTS:
                prof = dataclasses.replace(base, slots_per_week=slots)
                for mode, (fc, ac3) in MODES.items():
                    t0 = time.perf_counter()
                    r = schedule(plan, prof, use_fc=fc, use_ac3=ac3, max_assignments=LIMIT)
                    rows.append({"profile": base.student_id, "slots_per_week": slots, "mode": mode,
                                 "sessions": sum(plan), "ok": int(r.ok),
                                 "assignments": r.assignments, "backtracks": r.backtracks,
                                 "values_pruned": r.values_pruned,
                                 "hit_limit": int(any("stopped" in m for m in r.log)),
                                 "seconds": time.perf_counter() - t0})
        return rows
    return experiment


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--profiles", type=int, default=30)
    args = ap.parse_args(argv)
    df = harness.run("exp2_csp", make_experiment(args.real, args.profiles))
    summary = harness.summarize(df, ["slots_per_week", "mode"], METRICS, name="exp2_csp")
    order = {m: i for i, m in enumerate(MODES)}
    summary = summary.sort_values(["slots_per_week", "mode"], key=lambda c: c.map(order) if c.name == "mode" else c)
    print(summary.round(3).to_string(index=False))
    for metric, ylabel, log in [("assignments", "assignments (mean over 5 seeds)", True),
                                ("backtracks", "backtracks (mean over 5 seeds)", True),
                                ("seconds", "seconds per plan", False)]:
        plots.grouped_bars(summary, "slots_per_week", "mode", metric,
                           config.RESULTS_DIR / f"exp2_{metric}.png",
                           f"Experiment 2: {metric} by CSP mode", ylabel, log=log, series_labels=LABELS)
    print(f"saved tables and plots to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
