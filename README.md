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

```bash
python -m data.make_data           # download + preprocess OULAD (fixed seed)
python -m nn.train --seed 42       # train the NN, compare with the average baseline
python run_demo.py                 # integrated demo with plots
python -m eval.exp1_search         # Experiment 1: UCS vs Greedy vs A*
python -m eval.exp2_csp            # Experiment 2: BT vs BT+FC vs BT+FC+AC-3
python -m eval.exp3_nn_vs_avg      # Experiment 3: NN vs average predictor
pytest -q                          # tests
```

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

- **Person A (person 1): Abdo Kahrawi**, prediction side: data, `nn/`, `search/`, Experiments 1 and 3.
- **Person B (person 2): Zeyad Salem**, scheduling side: `csp/`, `integration/`, eval harness, Experiment 2, `run_demo.py`.

In the Owner column above, A means Abdo and B means Zeyad.

## Status

Skeleton only. Functions raise `NotImplementedError` with the owner and target week.
`common/stubs.py` provides a fake predictor and a simple scheduler so both sides can build in parallel.
