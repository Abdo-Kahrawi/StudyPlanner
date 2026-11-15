"""Project deliverable: planner output for 30+ student profiles. Owner: Person B.

Writes, for every profile:
    results/profiles_summary.csv    one row per student: plan, predicted marks, status, counters
    results/profiles_timetables.csv one row per scheduled session: student, assignment, week, slot

Usage:
    python -m eval.run_profiles               # synthetic profiles until real data exists
    python -m eval.run_profiles --real        # profiles from OULAD (data/processed/features.parquet)
    python -m eval.run_profiles --n 50 --predictor avg --strategy greedy
"""
from __future__ import annotations

import argparse

import pandas as pd

from common import config
from csp.solver import MODES
from eval import harness
from integration.planner import plan_for


def run_profiles(profiles, predictor, search_fn, strategy="ucs", csp="ac3"):
    fc, ac3 = MODES[csp]
    summary, sessions = [], []
    for p in profiles:
        out = plan_for(p, predictor, strategy, use_fc=fc, use_ac3=ac3, search_fn=search_fn)
        marks = out.predicted
        row = {"student_id": p.student_id, "target": p.target, "slots_per_week": p.slots_per_week,
               "status": out.search.status, "total_sessions": sum(out.plan) if out.plan else None,
               "nodes_expanded": out.search.nodes_expanded, "csp_calls": len(out.csp_calls),
               "nn_yes_csp_no": len(out.nn_yes_csp_no), "seconds": round(out.seconds, 4),
               "message": out.message}
        for i, a in enumerate(p.assignments):
            row[f"a{i + 1}_id"] = a.id
            row[f"a{i + 1}_deadline_week"] = a.deadline_week
            row[f"a{i + 1}_sessions"] = out.plan[i] if out.plan else None
            row[f"a{i + 1}_predicted"] = round(float(marks[i]), 2) if marks is not None else None
        summary.append(row)
        if out.schedule and out.schedule.ok:
            for (aid, k), (week, slot) in sorted(out.schedule.timetable.items(), key=lambda kv: kv[1]):
                sessions.append({"student_id": p.student_id, "assignment": aid, "session": k + 1,
                                 "week": week + 1, "slot": slot + 1})
    return pd.DataFrame(summary), pd.DataFrame(sessions)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--predictor", choices=["nn", "avg"], default="nn")
    ap.add_argument("--strategy", choices=["ucs", "greedy", "astar"], default="ucs")
    ap.add_argument("--csp", choices=list(MODES), default="ac3")
    args = ap.parse_args(argv)

    profiles = harness.load_profiles(args.n, args.seed, args.real)
    summary, sessions = run_profiles(profiles, harness.load_predictor(args.predictor),
                                     harness.load_search(), args.strategy, args.csp)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.RESULTS_DIR / "profiles_summary.csv", index=False)
    sessions.to_csv(config.RESULTS_DIR / "profiles_timetables.csv", index=False)
    found = (summary["status"] == "found").sum()
    print(f"{len(summary)} profiles: {found} plans found, {len(summary) - found} unreachable "
          f"at their capacity; {summary['nn_yes_csp_no'].gt(0).sum()} had an 'NN yes, CSP no' plan.")
    print(f"Mean sessions when found: {summary.loc[summary.status == 'found', 'total_sessions'].mean():.1f}")
    print(f"Saved {config.RESULTS_DIR / 'profiles_summary.csv'} and profiles_timetables.csv")


if __name__ == "__main__":
    main()
