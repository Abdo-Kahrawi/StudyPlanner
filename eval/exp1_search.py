"""Experiment 1: UCS vs Greedy vs A* (nodes expanded, cost).
P1 spec: "(1) UCS vs Greedy vs A* : nodes expanded and cost."
Common checklist: experiments need tables + plots, at least 5 runs/seeds, mean and std dev.

Uses ONLY public APIs:
    from search import StudyPlannerProblem, uniform_cost_search, greedy_search, astar_search
    from nn import StudyPredictor   # for real NN predictor; falls back to a dummy
                                     # linear predictor if nn_weights.pt is missing

Method:
  - Sample N_STUDENTS student profiles from train_data.csv (seeded).
  - For each student, take up to 3 assessments as the 3 assignments (spec example:
    target 55 for three assignments). Each assessment row is one profile dict.
  - Same problem (same profiles, same targets) is solved by all 3 algorithms.
    A fresh StudyPlannerProblem is built per algorithm so nn_calls / caches don't leak.
  - Metrics per run: success, cost (total sessions), nodes_expanded, nodes_generated,
    max_frontier_size, nn_calls, time.
  - Aggregate over seeds: mean +/- std per algorithm. UCS cost is the optimum;
    Greedy/A* cost is reported as excess over UCS (cost - UCS cost) on problems
    where all succeed.

Run:
    python -m eval.exp1_search --train_csv train_data.csv --n_students 30 --n_seeds 5 --target 55
    python -m eval.exp1_search --help

Outputs (in --out_dir, default eval/results/exp1/):
    exp1_raw.csv        one row per (seed, student, algorithm)
    exp1_summary.csv    mean +/- std per algorithm
    exp1_nodes.png      mean nodes expanded per algorithm
    exp1_cost.png       mean cost + excess over UCS
"""
from __future__ import annotations
import argparse, time, sys
from pathlib import Path
import numpy as np
import pandas as pd

# Only public APIs
try:
    from search import StudyPlannerProblem, uniform_cost_search, greedy_search, astar_search
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from search import StudyPlannerProblem, uniform_cost_search, greedy_search, astar_search

try:
    from nn import StudyPredictor
except Exception:
    StudyPredictor = None  # fallback below

ALGORITHMS = {
    "UCS": uniform_cost_search,
    "Greedy": greedy_search,
    "A*": astar_search,
}

class _DummyPredictor:
    """Fallback when no trained NN exists. Linear +2 marks/session, so search still
    demonstrates the comparison. Clearly labelled in output so it can't be mistaken
    for the real NN. The real run uses StudyPredictor."""
    name = "dummy-linear (+2/session)"
    def predict_score(self, profile, proposed_plan):
        base = profile.get("cumulative_avg_score") or profile.get("prev_score") or 40.0
        sessions = proposed_plan if isinstance(proposed_plan, int) else 0
        return float(max(0.0, min(100.0, base + 2.0 * sessions)))

def load_predictor(weights="nn_weights.pt", scaler="scaler.npz"):
    if StudyPredictor is not None and Path(weights).exists() and Path(scaler).exists():
        try:
            p = StudyPredictor(weights_path=str(weights), scaler_path=str(scaler))
            p.name = "NN (StudyPredictor)"
            return p
        except Exception as e:
            print(f"[exp1] NN load failed ({e}), using dummy predictor")
    print("[exp1] WARNING: no trained NN found - using dummy linear predictor. Train first: python -m nn.train")
    return _DummyPredictor()

def sample_problems(train_csv, n_students, target, seed, n_assignments=3, max_per=10, max_total=30):
    """One problem per student: up to n_assignments assessment rows for that student."""
    df = pd.read_csv(train_csv)
    rng = np.random.default_rng(seed)
    students = rng.choice(df["id_student"].unique(), size=min(n_students, df["id_student"].nunique()), replace=False)
    problems = []
    for sid in students:
        rows = df[df["id_student"] == sid].head(n_assignments)
        if len(rows) < 2:
            continue
        profiles = [r.to_dict() for _, r in rows.iterrows()]
        targets = [float(target)] * len(profiles)
        problems.append({"student": float(sid), "profiles": profiles, "targets": targets})
    return problems

def run_experiment(train_csv="train_data.csv", n_students=30, n_seeds=5, target=55.0,
                   out_dir="eval/results/exp1", weights="nn_weights.pt", scaler="scaler.npz",
                   max_per=10, max_total=30, seed0=42):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    predictor = load_predictor(weights, scaler)
    rows = []
    for s in range(n_seeds):
        seed = seed0 + s
        problems = sample_problems(train_csv, n_students, target, seed)
        for prob in problems:
            # Fresh problem per algorithm: no shared cache / counters
            results = {}
            for name, fn in ALGORITHMS.items():
                p = StudyPlannerProblem(prob["profiles"], prob["targets"], predictor=predictor,
                                        max_sessions_per_assignment=max_per, max_total_sessions=max_total)
                t0 = time.time()
                res = fn(p)
                dt = time.time() - t0
                results[name] = res
                rows.append({
                    "seed": seed, "student": prob["student"], "algorithm": name,
                    "success": bool(res.success),
                    "cost_total_sessions": res.cost if res.success else np.nan,
                    "allocation": list(res.solution_state) if res.solution_state else None,
                    "nodes_expanded": res.nodes_expanded,
                    "nodes_generated": res.nodes_generated,
                    "max_frontier_size": res.max_frontier_size,
                    "nn_calls": res.nn_calls,
                    "time_s": dt,
                    "predictor": getattr(predictor, "name", type(predictor).__name__),
                })
            # Excess over UCS optimum (only if UCS succeeded)
            ucs_cost = results["UCS"].cost if results["UCS"].success else np.nan
            for r in rows[-len(ALGORITHMS):]:
                r["excess_over_ucs"] = (r["cost_total_sessions"] - ucs_cost) if (r["success"] and np.isfinite(ucs_cost)) else np.nan
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "exp1_raw.csv", index=False)

    # Summary: mean +/- std per algorithm (spec: at least 5 runs, mean + std)
    summary = raw.groupby("algorithm").agg(
        n=("success", "size"),
        success_rate=("success", "mean"),
        cost_mean=("cost_total_sessions", "mean"), cost_std=("cost_total_sessions", "std"),
        excess_mean=("excess_over_ucs", "mean"), excess_std=("excess_over_ucs", "std"),
        nodes_expanded_mean=("nodes_expanded", "mean"), nodes_expanded_std=("nodes_expanded", "std"),
        nodes_generated_mean=("nodes_generated", "mean"), nodes_generated_std=("nodes_generated", "std"),
        nn_calls_mean=("nn_calls", "mean"), nn_calls_std=("nn_calls", "std"),
        time_mean=("time_s", "mean"), time_std=("time_s", "std"),
    ).reset_index()
    summary.to_csv(out / "exp1_summary.csv", index=False)

    print(f"\n=== Experiment 1: UCS vs Greedy vs A*  (predictor: {getattr(predictor,'name',type(predictor).__name__)}) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nUCS is optimal: Greedy/A* excess_over_ucs should be >= 0. A* should expand far fewer nodes than UCS if h is admissible.")
    print(f"Saved: {out/'exp1_raw.csv'}  {out/'exp1_summary.csv'}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = ["UCS", "Greedy", "A*"]
        s = summary.set_index("algorithm").reindex(order)
        fig, ax = plt.subplots()
        ax.bar(order, s["nodes_expanded_mean"], yerr=s["nodes_expanded_std"], capsize=5)
        ax.set_ylabel("Nodes expanded (mean +/- std)")
        ax.set_title("Exp1: Nodes expanded by algorithm")
        fig.tight_layout(); fig.savefig(out / "exp1_nodes.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots()
        ax.bar(order, s["cost_mean"], yerr=s["cost_std"], capsize=5)
        ax.set_ylabel("Total sessions / cost (mean +/- std)")
        ax.set_title("Exp1: Cost (total sessions) by algorithm")
        fig.tight_layout(); fig.savefig(out / "exp1_cost.png", dpi=150); plt.close(fig)
        print(f"Saved: {out/'exp1_nodes.png'}  {out/'exp1_cost.png'}")
    except Exception as e:
        print(f"[exp1] plots skipped: {e}")
    return raw, summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="train_data.csv")
    ap.add_argument("--n_students", type=int, default=30)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--target", type=float, default=55.0)
    ap.add_argument("--out_dir", default="eval/results/exp1")
    ap.add_argument("--weights", default="nn_weights.pt")
    ap.add_argument("--scaler", default="scaler.npz")
    ap.add_argument("--max_per", type=int, default=10)
    ap.add_argument("--max_total", type=int, default=30)
    args = ap.parse_args()
    run_experiment(args.train_csv, args.n_students, args.n_seeds, args.target,
                   args.out_dir, args.weights, args.scaler, args.max_per, args.max_total)
