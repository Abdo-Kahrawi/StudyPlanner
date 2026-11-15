# Results on real OULAD data

Run on 2026-11-15 from the commands in the README (course BBB; train 2013B/2013J/2014B,
test 2014J; seed 42 for data and training; experiments over seeds 0-4 with 30 profiles each).
Mean (std) over seeds. Raw and summary CSVs and plots are written to `results/`.

## Data

| | Rows | Mean mark |
| --- | --- | --- |
| train (2013B, 2013J, 2014B) | 35 601 | 78.9 |
| test (2014J) | 7 399 | 65.7 |

2014J is a different cohort with a different assessment set: 5 TMAs and no CMAs (the train
presentations also have high-scoring CMAs), and lower marks. This shift explains most of the gap
between validation RMSE (13.7) and test RMSE (~29).

## Neural network

| Model | Test RMSE | Test MAE | Val RMSE | Monotonicity violations* | Mark gain, 0 -> 10 sessions* |
| --- | --- | --- | --- | --- | --- |
| Mean-mark baseline | 30.1 | | | | |
| Previous-average baseline | 38.4 | 30.3 | | | 0 (ignores effort) |
| Plain MLP (`--arch mlp`) | **28.9** | | 13.7 | 39% of steps, 57% of curves | +1.0 (59% of curves move < 1 mark) |
| Monotone NN (`--arch monotone`, used by the planner) | 29.8 | 22.2 | 13.7 | **0%** | **+7.7** (max +2.7 per session) |

\* `python -m eval.monotonicity --real`: 30 test profiles x 3 assignments, 0-10 sessions each.

Reading: clicks only weakly predict marks in BBB (correlation 0.10 between 4-week clicks and
mark). The plain MLP learns almost nothing from them, and adding sessions often LOWERS its
prediction slightly, so search has nothing to optimise. The monotone NN (the fallback planned in
the risk table) splits the model into an MLP on the non-click features plus a click branch with
positive weights on log(1 + clicks): it can only raise the mark, gives diminishing returns, and
costs 0.9 RMSE on the test cohort. Both beat the two baselines.

## Experiment 1: UCS vs Greedy vs A* (`python -m eval.exp1_planner --real`)

NN predictor, BT+FC+AC-3, 3 slots per week. Targets set to 60 / 70 / 80 because most real
students already reach their generated target (50-70) with 0 sessions.

| Target | Solved | Cost (sessions, solved) | Nodes expanded, solved: UCS / Greedy / A* | Excess cost vs UCS |
| --- | --- | --- | --- | --- |
| 60 | 71% (12) | 1.0 (0.6) | 23.8 / 3.4 / 7.2 | 0 / 0 |
| 70 | 60% (11) | 0.4 (0.2) | 5.2 / 1.8 / 1.9 | 0 / 0 |
| 80 | 25% (8) | 6.7 (0.7) | 124.9 / 8.9 / 34.5 | 0 / 0 |

Unsolved profiles expand the whole capped space (1 331 plans, up to 10 sessions per assignment)
with every strategy. A* (admissible heuristic: remaining deficit / largest one-session gain)
expands 3.6x fewer nodes than UCS at target 80 with the same optimal cost. Greedy expands the
fewest and here also found optimal plans, but it has no optimality guarantee.

## Experiment 2: BT vs BT+FC vs BT+FC+AC-3 (`python -m eval.exp2_csp --real`)

Plans from the Experiment 1 setting (UCS, 3 slots), each scheduled at 2, 3 and 4 slots per week.

| Slots/week | Mode | Assignments | Backtracks | Values pruned | Schedulable |
| --- | --- | --- | --- | --- | --- |
| 2 | BT | 19.3 (1.8) | 11.7 (2.2) | 0 | 81% |
| 2 | BT+FC | 19.3 (1.8) | 11.7 (2.2) | 129 (22) | 81% |
| 2 | BT+FC+AC-3 | **7.6 (0.4)** | **0** | 70 (4) | 81% |
| 3 and 4 | all modes | 9.7 (0.3) | 0 | 0 / 47 / 86 | 100% |

With 2 slots, AC-3 finds every infeasible plan before search starts (0 backtracks).
With 3-4 slots every plan fits on the first try and pruning only costs time.

## Experiment 3: NN vs average predictor in the planner (`python -m eval.exp3_planner --real`)

UCS, BT+FC+AC-3, 3 slots. "Call correct" = the predictor's "0 sessions are enough" matches the
student's ACTUAL 2014J marks.

| Target | Predictor | Solved | Cost (solved) | Call correct |
| --- | --- | --- | --- | --- |
| 60 | average | 60% | 0 (never plans sessions) | 74% |
| 60 | plain MLP | 67% | 0.2 | 68% |
| 60 | monotone NN | **71%** | 1.0 | 74% |
| 70 | average | 60% | 0 | 53% |
| 70 | plain MLP | 50% | 0.6 | 65% |
| 70 | monotone NN | 60% | 0.4 | 59% |
| 80 | average | 0% | - | 95% |
| 80 | plain MLP | 0% | - | 95% |
| 80 | monotone NN | **25%** | 6.7 | 94% |

The average predictor ignores effort: it says either "0 sessions" or "unreachable", never a plan
with sessions. Only the effort-sensitive NN produces plans that add sessions. Standalone, the NN
beats the average predictor (RMSE 29.8 vs 38.4, `python -m eval.exp3_nn_vs_avg`).

## Planner on 30 profiles at their own generated targets (`python -m eval.run_profiles --real`)

21 of 30 plans found (mean 0.6 sessions: most students are already predicted above target),
9 unreachable within 10 sessions per assignment. The "NN yes, CSP no" case is shown with
`python run_demo.py --profile tight` (280 plans the NN accepts are rejected by the CSP).

## Limits of these numbers

- Clicks and marks are observed together. More clicks may reflect stronger or more motivated
  students; the planner shows what-if predictions, not guaranteed outcomes.
- The effort rule (`nn/effort.py`) and clicks_per_session (10 clicks, from the generation rule)
  are assumptions, not measurements.
- One course (BBB) and one test presentation.
