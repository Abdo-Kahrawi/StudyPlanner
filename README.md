# StudyPlanner: Assessment-Aware Study Planner

A decision-support planner for distance-learning students. It decides how many study sessions to give
each upcoming assignment, and in which weeks, so the predicted marks reach a target with the fewest sessions.

- **Neural network** predicts an assignment mark from earlier marks and recent online activity (clicks).
- **Search** (UCS, Greedy Best-First, A*) adds one session at a time to find the smallest plan that meets the target.
- **CSP** (Backtracking, Forward Checking, AC-3) places the sessions into weekly time slots before each deadline.

If no plan works, the planner reports that the target is unreachable at this capacity and shows the best predicted marks.

> This is a "what-if" tool. The link between online activity and marks in the data is observational,
> not proof of cause. See the limitations section of the report.

Dataset: [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset),
CC BY 4.0 ([figshare copy](https://figshare.com/articles/OULAD_Open_University_Learning_Analytics_Dataset/5081998)).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Put the OULAD CSVs in `data/raw/` (or let step 1 download them), then from the repo root:

```bash
python -m data.make_data                  # 1. raw OULAD -> data/processed/{train,test}_data.csv (seed 42)
python -m nn.train --seed 42              # 2. train the NN (monotone in clicks), compare with baselines -> nn/weights/
python -m eval.monotonicity --real        # 3. monotonicity check and violation report
python run_demo.py                        # 4. integrated demo with plots (see docs/demo_script.md)
python -m eval.run_profiles --real        # 5. planner output for 30 OULAD profiles
python -m eval.exp1_planner --real        # 6. Experiment 1: UCS vs Greedy vs A* in the planner
python -m eval.exp2_csp --real            # 7. Experiment 2: BT vs BT+FC vs BT+FC+AC-3
python -m eval.exp3_planner --real        # 8. Experiment 3: NN vs average predictor in the planner
python -m eval.exp3_nn_vs_avg             #    + standalone NN vs average RMSE / MAE
python -m eval.exp1_search                #    + standalone search comparison on training rows
pytest -q                                 # tests
```

Results (CSV tables and PNG plots) go to `results/`; the numbers from the real-data run are in
`docs/results.md`. For the plain MLP used as a comparison in Experiment 3, also run
`python -m nn.train --arch mlp --save_path nn/weights/mlp/nn_weights.pt --scaler_path nn/weights/mlp/scaler.npz --log_path nn/weights/mlp/train_log.json --comparison_path nn/weights/mlp/baseline_comparison.txt`. Without trained weights, the demo and
experiments fall back to the stub predictor and print a note saying so.

## Layout and owners

| Module | Owner | Purpose |
| --- | --- | --- |
| `common/` | A + B | Shared types (the contract), config, stubs |
| `data/` | A | `make_data.py`, feature schema |
| `data/profiles.py` | B | 30+ student profiles, generated capacity/target/session length |
| `nn/` | A | Mark predictor, baseline, effort-to-clicks rule |
| `search/` | A | UCS, Greedy, A* with node counters |
| `csp/` | B | Backtracking, Forward Checking, AC-3 with counters |
| `integration/` | B | The search / NN / CSP loop |
| `eval/` | B harness, A exp 1 and 3, B exp 2 | Experiments, mean and std over 5 seeds |
| `tests/` | each owner | Unit tests + 5 normal / 3 edge / 2 impossible planner cases |
| `run_demo.py` | B | Integrated demo |

## Team

- **Abdo Kahrawi**, prediction side: data, `nn/`, `search/`, Experiments 1 and 3.
- **Zeyad Salem**, scheduling side: `csp/`, `integration/`, eval harness, Experiment 2, `run_demo.py`.

In the Owner column above, A means Abdo and B means Zeyad.

## Status

All modules are implemented. Run `python -m data.make_data` then `python -m nn.train` before the
demo and experiments: until trained weights exist in `nn/weights/`, the planner falls back to the
stub predictor in `common/stubs.py` and says so.
