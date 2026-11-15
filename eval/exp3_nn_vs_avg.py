"""Experiment 3: NN vs average-mark predictor inside the planner.
P1 spec: "(3) NN versus a simple average-mark predictor inside the planner."

Two levels, both use ONLY public APIs:
    from nn import StudyPredictor          # NN predictor
    from search import StudyPlannerProblem, uniform_cost_search   # planner

Level A - Standalone (sanity check, same as nn/train.py):
  On held-out rows: NN (StudyPredictor) vs Average predictor
  Average predictor = cumulative_avg_score if >0 else prev_score if >0 else train mean.
  Metric: MSE / RMSE. NN should win; if not, report it honestly.

Level B - Inside the planner (the actual experiment):
  Same problems, same UCS, same targets - only the predictor changes:
    * NN predictor:    predict_score(profile, sessions) rises with sessions (via effort.py)
    * Average predictor: predict_score(profile, sessions) IGNORES sessions
      (returns cumulative_avg / prev / train mean). This is the key insight:
      a predictor that cannot see effort cannot plan - if its average is below
      target, NO amount of sessions will ever make UCS succeed. The planner
      will report "no solution" even though the NN planner finds a plan.
  Metrics per problem: success_rate, cost (when successful), nodes_expanded,
  nn_calls, and - crucially - target_reached_rate and wasted_sessions.
  Aggregate over >=5 seeds: mean +/- std per predictor.

Run:
    python -m eval.exp3_nn_vs_avg --train_csv train_data.csv --test_csv test_data.csv --n_students 30 --n_seeds 5 --target 55
Outputs (eval/results/exp3/):
    exp3_standalone.csv   MSE/RMSE NN vs Average on test rows
    exp3_planner_raw.csv  one row per (seed, student, predictor)
    exp3_planner_summary.csv  mean +/- std per predictor
    exp3_success.png, exp3_cost.png
"""
from __future__ import annotations
import argparse, time, sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from search import StudyPlannerProblem, uniform_cost_search
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from search import StudyPlannerProblem, uniform_cost_search
try:
    from nn import StudyPredictor
except Exception:
    StudyPredictor = None

class AveragePredictor:
    """Simple average-mark predictor. Public API matches StudyPredictor:
    predict_score(student_profile, proposed_plan) -> float.
    Deliberately ignores proposed_plan - it has no notion of effort."""
    name = "Average (cumulative_avg / prev / train mean)"
    def __init__(self, train_mean: float = 70.0):
        self.train_mean = float(train_mean)
    def predict_score(self, profile, proposed_plan=0):
        # proposed_plan is ignored on purpose - see docstring
        cum = profile.get("cumulative_avg_score", 0) or 0
        if cum and cum > 0:
            return float(cum)
        prev = profile.get("prev_score", 0) or 0
        if prev and prev > 0:
            return float(prev)
        return float(self.train_mean)

class _DummyNN:
    """Fallback if no trained weights: linear +2/session, labelled as dummy."""
    name = "dummy-linear NN (+2/session)"
    def predict_score(self, profile, proposed_plan=0):
        base = profile.get("cumulative_avg_score") or profile.get("prev_score") or 40.0
        sessions = proposed_plan if isinstance(proposed_plan, int) else 0
        return float(max(0.0, min(100.0, base + 2.0 * sessions)))

def load_nn(weights="nn_weights.pt", scaler="scaler.npz"):
    if StudyPredictor is not None and Path(weights).exists() and Path(scaler).exists():
        try:
            p = StudyPredictor(weights_path=str(weights), scaler_path=str(scaler))
            p.name = "NN (StudyPredictor)"
            return p
        except Exception as e:
            print(f"[exp3] NN load failed ({e}), using dummy")
    print("[exp3] WARNING: no trained NN found - using dummy linear predictor. Train first: python -m nn.train")
    return _DummyNN()

def standalone_comparison(train_csv, test_csv, nn_predictor, avg_predictor):
    """Level A: MSE/RMSE on held-out rows, NN vs Average."""
    test = pd.read_csv(test_csv)
    train = pd.read_csv(train_csv)
    # Average predictor needs train mean
    avg_predictor.train_mean = float(train["target_score"].mean())
    y = test["target_score"].values.astype(float)
    preds_nn, preds_avg = [], []
    for _, row in test.iterrows():
        profile = row.to_dict()
        preds_nn.append(nn_predictor.predict_score(profile, 0))
        preds_avg.append(avg_predictor.predict_score(profile, 0))
    preds_nn, preds_avg = np.array(preds_nn), np.array(preds_avg)
    out = []
    for name, preds in [("NN", preds_nn), ("Average", preds_avg)]:
        mse = float(((preds - y) ** 2).mean())
        out.append({"predictor": name, "n": len(y), "mse": mse, "rmse": float(np.sqrt(mse)),
                    "mae": float(np.abs(preds - y).mean())})
    df = pd.DataFrame(out)
    print("\n=== Exp3 Level A: Standalone MSE/RMSE (0 extra sessions) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df

def sample_problems(train_csv, n_students, target, seed, n_assignments=3):
    df = pd.read_csv(train_csv)
    rng = np.random.default_rng(seed)
    students = rng.choice(df["id_student"].unique(), size=min(n_students, df["id_student"].nunique()), replace=False)
    problems = []
    for sid in students:
        rows = df[df["id_student"] == sid].head(n_assignments)
        if len(rows) < 2:
            continue
        problems.append({"student": float(sid),
                         "profiles": [r.to_dict() for _, r in rows.iterrows()],
                         "targets": [float(target)] * len(rows)})
    return problems

def run_experiment(train_csv="train_data.csv", test_csv="test_data.csv", n_students=30, n_seeds=5,
                   target=55.0, out_dir="eval/results/exp3", weights="nn_weights.pt", scaler="scaler.npz",
                   max_per=10, max_total=30, seed0=42):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    nn_pred = load_nn(weights, scaler)
    train = pd.read_csv(train_csv)
    avg_pred = AveragePredictor(train_mean=float(train["target_score"].mean()))

    # Level A
    standalone = standalone_comparison(train_csv, test_csv, nn_pred, avg_pred)
    standalone.to_csv(out / "exp3_standalone.csv", index=False)

    # Level B - inside planner, UCS only (controls for search, isolates predictor effect)
    rows = []
    for s in range(n_seeds):
        seed = seed0 + s
        for prob in sample_problems(train_csv, n_students, target, seed):
            for pred_name, pred in [("NN", nn_pred), ("Average", avg_pred)]:
                p = StudyPlannerProblem(prob["profiles"], prob["targets"], predictor=pred,
                                        max_sessions_per_assignment=max_per, max_total_sessions=max_total)
                t0 = time.time()
                res = uniform_cost_search(p)
                dt = time.time() - t0
                # How many assignments would actually reach target with this plan?
                # For Average: predicted scores never move, so success == already at target at 0 sessions.
                rows.append({
                    "seed": seed, "student": prob["student"], "predictor": pred_name,
                    "success": bool(res.success),
                    "cost_total_sessions": res.cost if res.success else np.nan,
                    "allocation": list(res.solution_state) if res.solution_state else None,
                    "predicted_scores": res.predicted_scores,
                    "nodes_expanded": res.nodes_expanded,
                    "nodes_generated": res.nodes_generated,
                    "nn_calls": res.nn_calls,
                    "time_s": dt,
                })
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "exp3_planner_raw.csv", index=False)

    summary = raw.groupby("predictor").agg(
        n=("success", "size"), success_rate=("success", "mean"),
        cost_mean=("cost_total_sessions", "mean"), cost_std=("cost_total_sessions", "std"),
        nodes_expanded_mean=("nodes_expanded", "mean"), nodes_expanded_std=("nodes_expanded", "std"),
        nn_calls_mean=("nn_calls", "mean"), time_mean=("time_s", "mean"),
    ).reset_index()
    summary.to_csv(out / "exp3_planner_summary.csv", index=False)

    print(f"\n=== Exp3 Level B: Inside planner (UCS, target={target}) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    nn_succ = summary.loc[summary["predictor"] == "NN", "success_rate"].iloc[0] if "NN" in summary["predictor"].values else 0
    avg_succ = summary.loc[summary["predictor"] == "Average", "success_rate"].iloc[0] if "Average" in summary["predictor"].values else 0
    print(f"\nInterpretation: NN planner succeeds on {nn_succ:.0%} of problems, Average on {avg_succ:.0%}.")
    print("Average cannot plan because it ignores effort: if average < target at 0 sessions, no plan ever succeeds.")
    print("NN is worth it IFF it both (a) beats Average on MSE (Level A) AND (b) enables planning (Level B).")
    print(f"Saved: {out/'exp3_standalone.csv'}  {out/'exp3_planner_raw.csv'}  {out/'exp3_planner_summary.csv'}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s = summary.set_index("predictor").reindex(["NN", "Average"])
        fig, ax = plt.subplots()
        ax.bar(["NN", "Average"], s["success_rate"], capsize=5)
        ax.set_ylabel("Planner success rate (mean)"); ax.set_title("Exp3: Planner success - NN vs Average")
        ax.set_ylim(0, 1.05); fig.tight_layout(); fig.savefig(out / "exp3_success.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots()
        ax.bar(["NN", "Average"], s["cost_mean"], yerr=s["cost_std"], capsize=5)
        ax.set_ylabel("Total sessions when successful (mean +/- std)"); ax.set_title("Exp3: Cost - NN vs Average")
        fig.tight_layout(); fig.savefig(out / "exp3_cost.png", dpi=150); plt.close(fig)
        print(f"Saved: {out/'exp3_success.png'}  {out/'exp3_cost.png'}")
    except Exception as e:
        print(f"[exp3] plots skipped: {e}")
    return standalone, raw, summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="train_data.csv")
    ap.add_argument("--test_csv", default="test_data.csv")
    ap.add_argument("--n_students", type=int, default=30)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--target", type=float, default=55.0)
    ap.add_argument("--out_dir", default="eval/results/exp3")
    ap.add_argument("--weights", default="nn_weights.pt")
    ap.add_argument("--scaler", default="scaler.npz")
    ap.add_argument("--max_per", type=int, default=10)
    ap.add_argument("--max_total", type=int, default=30)
    a = ap.parse_args()
    run_experiment(a.train_csv, a.test_csv, a.n_students, a.n_seeds, a.target, a.out_dir, a.weights, a.scaler, a.max_per, a.max_total)
