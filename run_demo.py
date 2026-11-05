"""Integrated demo: input, solution, and plots of how search, NN and CSP cooperate. Owner: Person B.

Examples:
    python run_demo.py                                  # the worked example from the brief
    python run_demo.py --profile tight                  # NN says "enough" but the CSP rejects
    python run_demo.py --profile impossible             # target 100: clear "no solution"
    python run_demo.py --strategy greedy --csp bt --trace
    python run_demo.py --predictor avg                  # NN off: average-mark predictor
    python run_demo.py --compare-predictors             # NN on vs off, side by side
    python run_demo.py --profile 7 --real               # profile 7 built from OULAD
"""
from __future__ import annotations

import argparse
import dataclasses

from common import config
from common.examples import example_profile
from common.types import Assignment
from csp.solver import MODES, schedule
from eval import harness
from integration.planner import format_timetable, plan_for

CSP_LABELS = {"bt": "BT", "fc": "BT+FC", "ac3": "BT+FC+AC-3"}


def get_profile(name: str, real: bool):
    if name == "example":
        return example_profile()
    if name == "tight":      # A1 needs 5 sessions but only 4 slots exist before its deadline
        p = example_profile(slots_per_week=2, target=65.0)
        return dataclasses.replace(p, assignments=(Assignment("A1", "TMA", 20.0, 2), *p.assignments[1:]))
    if name == "impossible":
        return example_profile(target=100.0)
    return harness.load_profiles(max(30, int(name) + 1), config.SEED, real)[int(name)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="example", help="example | tight | impossible | <index>")
    p.add_argument("--real", action="store_true", help="numbered profiles come from OULAD")
    p.add_argument("--strategy", choices=["ucs", "greedy", "astar"], default="ucs")
    p.add_argument("--csp", choices=list(MODES), default="ac3",
                   help="bt = backtracking only, fc = +forward checking, ac3 = +FC +AC-3")
    p.add_argument("--predictor", choices=["nn", "avg"], default="nn")
    p.add_argument("--compare-predictors", action="store_true", help="run with NN and with average")
    p.add_argument("--trace", action="store_true", help="print the search trace")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def show_input(profile):
    print("=" * 64)
    print(f"Student {profile.student_id}: target {profile.target:.0f}, "
          f"{profile.slots_per_week} free study slots per week, planning from week {profile.start_week + 1}")
    for a in profile.assignments:
        print(f"  {a.id}: {a.kind}, weight {a.weight:.0f}, deadline end of week {a.deadline_week}")
    print("=" * 64)


def show_output(out, trace: bool):
    s = out.search
    print(out.message)
    print(f"Search: {s.nodes_expanded} nodes expanded, {s.nodes_generated} generated")
    rej = out.nn_yes_csp_no
    print(f"CSP: called on {len(out.csp_calls)} plans; NN said enough but CSP rejected "
          f"{len(rej)} plan(s)" + (f", first: {rej[:3]}" if rej else ""))
    print("\nTimetable")
    print(format_timetable(out))
    if trace:
        print("\nSearch trace (first 40 steps)")
        for t in s.trace[:40]:
            print("  ", t)


def csp_comparison(out):
    """Schedule the most informative plan with each CSP mode, pre-check off."""
    plan = out.nn_yes_csp_no[0] if out.nn_yes_csp_no else out.plan
    rows = []
    for mode, (fc, ac3) in MODES.items():
        r = schedule(plan, out.profile, use_fc=fc, use_ac3=ac3)
        rows.append((mode, r))
    print(f"\nCSP on plan {plan} with and without FC / AC-3")
    print(f"  {'mode':<12}{'ok':<6}{'assignments':>12}{'backtracks':>12}{'pruned':>10}")
    for mode, r in rows:
        print(f"  {CSP_LABELS[mode]:<12}{str(r.ok):<6}{r.assignments:>12}{r.backtracks:>12}{r.values_pruned:>10}")
    return plan, rows


def plot(out, plan, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    prof = out.profile
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]                          # 1. search trace
    tr = out.search.trace
    if tr:
        margins = [t.get("min_margin", np.nan) for t in tr]
        ax.plot(range(1, len(tr) + 1), margins, lw=1, color="#4C72B0")
        ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_title("1. Search: lowest (predicted mark - target) per expanded plan")
    ax.set_xlabel("nodes expanded")
    ax.set_ylabel("margin to target")

    ax = axes[0, 1]                          # 2. NN marks for the plan
    marks = out.predicted
    ids = [a.id for a in prof.assignments]
    if marks is not None:
        bars = ax.bar(ids, marks, color="#55A868")
        for b, n in zip(bars, out.plan):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{n} sessions", ha="center", fontsize=9)
    ax.axhline(prof.target, color="#C44E52", ls="--", label=f"target {prof.target:.0f}")
    ax.set_ylim(0, 105)
    ax.set_title("2. Neural network: predicted marks for the chosen plan")
    ax.legend(frameon=False)

    ax = axes[1, 0]                          # 3. CSP pruning by mode
    labels = [CSP_LABELS[m] for m, _ in rows]
    x = np.arange(len(rows))
    for off, key, color in [(-0.2, "assignments", "#4C72B0"), (0.2, "backtracks", "#DD8452")]:
        vals = [getattr(r, key) for _, r in rows]
        ax.bar(x + off, vals, 0.4, label=key, color=color)
        for xi, v in zip(x, vals):
            ax.text(xi + off, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [f"{l}\n{'scheduled' if r.ok else 'rejected'}, {r.values_pruned} pruned"
                      for l, (_, r) in zip(labels, rows)])
    ax.set_yscale("symlog")
    ax.set_ylim(0, max(10, max(max(r.assignments, r.backtracks) for _, r in rows)) * 4)
    ax.set_title(f"3. CSP on plan {plan}: work with and without FC / AC-3")
    ax.legend(frameon=False)

    ax = axes[1, 1]                          # 4. timetable grid
    weeks = max(a.deadline_week for a in prof.assignments)
    grid = np.full((prof.slots_per_week, weeks), np.nan)
    if out.schedule and out.schedule.timetable:
        idx = {a.id: i for i, a in enumerate(prof.assignments)}
        for (aid, _), (w, s) in out.schedule.timetable.items():
            grid[s, w] = idx[aid]
    ax.imshow(grid, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#4C72B0", "#DD8452", "#55A868"]),
              vmin=0, vmax=2)
    for i, a in enumerate(prof.assignments):
        ax.axvline(a.deadline_week - 0.5, color="black", lw=1)
        ax.text(a.deadline_week - 0.6, prof.slots_per_week - 0.6, f"{a.id} due", ha="right",
                va="bottom", fontsize=8)
    ax.set_xticks(range(weeks), [str(w + 1) for w in range(weeks)])
    ax.set_yticks(range(prof.slots_per_week), [f"slot {s + 1}" for s in range(prof.slots_per_week)])
    ax.set_xlabel("week")
    ax.set_title("4. CSP timetable (colour = assignment)")

    fig.suptitle(out.message, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {path}")


def main(argv=None):
    args = parse_args(argv)
    profile = get_profile(args.profile, args.real)
    search_fn = harness.load_search()
    fc, ac3 = MODES[args.csp]
    show_input(profile)

    kinds = ["nn", "avg"] if args.compare_predictors else [args.predictor]
    for kind in kinds:
        predictor = harness.load_predictor(kind)
        print(f"\n--- predictor: {kind} ({predictor.name}), search: {args.strategy}, csp: {CSP_LABELS[args.csp]}")
        out = plan_for(profile, predictor, args.strategy, use_fc=fc, use_ac3=ac3, search_fn=search_fn)
        show_output(out, args.trace)
        plan, rows = csp_comparison(out)
        if not args.no_plots:
            config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            plot(out, plan, rows, config.RESULTS_DIR / f"demo_{args.profile}_{kind}.png")


if __name__ == "__main__":
    main()
