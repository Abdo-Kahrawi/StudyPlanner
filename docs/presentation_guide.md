# StudyPlanner: presentation guide

This guide explains every slide of the final presentation and every number on the web app, split by
who presents it. For each slide it gives what to say, where each number comes from (file and
command), and questions the TA may ask with short answers.

- **Abdo Kahrawi (prediction side):** data, neural network, search, Experiments 1 and 3, limits.
- **Zeyad Salem (scheduling side):** problem and loop, CSP, Experiment 2, integration,
  conclusions, the web app.
- **Both:** the live demo.

Total time: about 10 minutes of slides, then a 5 minute live demo.

| Time | Slide | Presenter |
| --- | --- | --- |
| 0:00 | 1 Cover, 2 Problem, 3 How it works | Zeyad |
| 1:30 | 4 Data, 5 Neural network, 6 Why monotone | Abdo |
| 3:30 | 7 Search, 8 Experiment 1 | Abdo |
| 5:00 | 9 CSP, 10 Experiment 2 | Zeyad |
| 6:30 | 11 Integration | Zeyad |
| 7:30 | 12 Experiment 3 | Abdo |
| 8:30 | 13 Limits | Abdo |
| 9:15 | 14 Conclusions | Zeyad |
| 10:00 | 15 to 19 The web app | Zeyad (19 Detail tabs: Abdo) |
| | 20 Live demo | Zeyad runs 1 and 2, Abdo runs 3 and 4 |

---

## The idea in one paragraph (both should know this)

A distance-learning student has three assignments with deadlines, a few free study slots each week
and a target mark. The planner answers two questions: how many study sessions should each
assignment get, and in which week and slot. Three techniques work together:

1. **Search** (UCS, Greedy or A*) proposes plans, adding one session at a time from (0, 0, 0).
2. A **neural network** predicts the mark of each assignment for that plan.
3. A **CSP** checks that the sessions fit into free slots before each deadline.

The answer is the plan with the fewest sessions where every predicted mark reaches the target and
the CSP can place it. If none exists, the planner says the target is unreachable at this capacity
and shows the closest plan. It is decision support: predictions, not promises.

Where the code lives:

| Folder | Owner | What it does |
| --- | --- | --- |
| `data/make_data.py` | Abdo | OULAD raw CSVs to train and test tables |
| `data/profiles.py` | Zeyad | Builds student profiles (demo students and 30 real test profiles) |
| `nn/` | Abdo | Monotone network, training, the effort rule (sessions to clicks) |
| `search/` | Abdo | UCS, Greedy, A*; `search/adapter.py` is the entry point the planner calls |
| `csp/` | Zeyad | Variables, BT, FC, AC-3, `schedule()` |
| `integration/planner.py` | Zeyad | `plan_for()`: the search, NN and CSP loop |
| `eval/` | Zeyad harness and Exp 2; Abdo Exp 1 and 3 | Experiments, plots, monotonicity check |
| `run_demo.py` | Zeyad | Command-line demo |
| `webapp/` | Zeyad | The browser interface for the TA |
| `common/types.py` | Both | The shared contract (StudentProfile, Plan, SearchResult, Predictor) |

---

# Part 1: Zeyad

## Slide 1. Cover (0:00)

**Say:** We built a planner that tells a distance-learning student how many study sessions to give
each assignment, and in which week, so the predicted marks reach a target with the fewest sessions.
A neural network predicts marks, search chooses how many sessions, and a CSP places them in free
slots. The data is real: the Open University dataset (OULAD), course BBB.

## Slide 2. The problem (0:00 to 1:30)

**Say:** The input is the brief's example. Target 55 on three assignments (TMAs) due at the end of
weeks 4, 8 and 12. Three free slots a week. The student averaged 30 on earlier marks and has no
recent online activity. The output is the number of sessions per assignment, a week-by-week
timetable and the predicted marks.

**Numbers on the slide**

| Number | Meaning | Where it comes from |
| --- | --- | --- |
| 55, weeks 4/8/12, 3 slots, average 30 | The "Example" student | `run_demo.py` `get_profile("example")`, built in `data/profiles.py` |
| (3, 1, 0), 4 sessions | The plan: 3 sessions for A1, 1 for A2, 0 for A3 | `python run_demo.py` or the web app, Example preset |
| 56.1, 56.2, 56.7 | Predicted marks with the plan | NN `predict(profile, (3,1,0))` |
| 47.4, 53.9, 56.7 (in the notes) | Predicted marks with no sessions | NN `predict(profile, (0,0,0))` |

**Explain:** A3 already passes with 0 sessions (56.7), so it gets none. A1 starts at 47.4 and needs
3 sessions to pass 55. A2 starts at 53.9 and 1 session is enough.

**Q: Why does A3 start higher than A1 with the same student?** The NN also uses each assignment's
weight and deadline week, which differ, so the starting marks differ.

## Slide 3. How it works (still before 1:30)

**Say:** Search owns the loop. It starts from (0, 0, 0) and each step adds one session to one
assignment. For each plan it asks the NN for the three marks. Only when every mark reaches the
target does it call the CSP, because the CSP is the slower check. If the CSP fits the plan, that is
the answer. If not, the plan is logged as "NN yes, CSP no" and search continues. CSP answers are
cached, so a plan is never scheduled twice.

**Number:** 1,331 = 11 x 11 x 11, every plan with 0 to 10 sessions per assignment (cap set in
`common/config.py`). When no plan works, search has tried all of them.

**Q: Why not run the CSP on every plan?** The NN check is cheap and rejects most plans. The CSP is
only worth running on plans that already reach the target (or that could become the closest
fallback plan).

## Slide 9. CSP (5:00 to 6:30)

**Say:** The CSP answers one question: can this plan fit in the student's week?

- **Variables:** one per planned session, for example A1-1, A1-2, A2-1 (`csp/variables.py`).
- **Domains:** every (week, slot) from the first week up to the week before that assignment's
  deadline. So the deadline is built into the domain, not a separate constraint.
- **Constraints:** all-different (no two sessions in the same week and slot), and order (session k
  of an assignment comes before session k+1, which also removes symmetric solutions).

Three solvers share the same code (`csp/solver.py`, `MODES`):

| Mode | Flag | What it does |
| --- | --- | --- |
| BT | `--csp bt` | Plain backtracking (`csp/backtracking.py`) |
| BT + FC | `--csp fc` | After each assignment, removes clashing values from neighbours (`csp/forward_checking.py`) |
| BT + FC + AC-3 | `--csp ac3` | Makes every arc consistent before search, then BT + FC (`csp/ac3.py`). Used by the planner. |

The planner also runs a quick counting check first (`capacity_ok`): if more sessions are due by a
deadline than there are slots before it, the plan fails immediately without search.

**Q: What is an arc here?** A pair of variables linked by a constraint. AC-3 removes a value from
one variable if no value in the other variable is compatible with it, and repeats until nothing
changes.

**Q: Why is FC not enough?** FC only looks one step ahead from the variable just assigned. AC-3
checks every pair before search, so it sees dead ends that FC only discovers after many steps.

## Slide 10. Experiment 2: CSP solvers (ends at 6:30)

**Setup:** 278 plans taken from the planner runs at targets 60, 70 and 80, each scheduled at 2, 3
and 4 slots a week by all three solvers, with the counting check turned off so the three do the same
work. Command: `python -m eval.exp2_csp --real`. Output: `results/exp2_csp_summary.csv`, plots
`results/exp2_*.png`, summary in `docs/results.md`.

| Number | Meaning |
| --- | --- |
| 11.7 backtracks (BT and BT+FC) | Average backtracks per plan at 2 slots a week |
| 129 values pruned | FC removed 129 values per plan on average but still backtracked as often |
| 0 backtracks, 70 pruned (AC-3) | AC-3 removes dead ends before search, so no backtracking is needed |
| 19.3 vs 7.6 | Variable assignments per plan: BT vs AC-3 |
| 81% | Share of plans that fit at 2 slots a week; at 3 and 4 slots, 100% fit with 0 backtracks |

**Say:** When slots are tight, AC-3 removes all backtracking. When slots are plentiful, all three
solvers are equally easy. The CSP solvers tab in the web app shows these same counters for one plan.

## Slide 11. Integration: NN yes, CSP no (6:30 to 7:30)

**Say:** The "tight" student: target 65, 2 slots a week, A1 due at the end of week 2. So only
2 x 2 = 4 slots exist for A1. The NN says A1 needs 6 sessions to reach 65. Search finds plans that
the NN accepts but the CSP rejects, for example (6, 4, 3), (7, 4, 3), (6, 5, 3). Because the CSP is
inside the goal test, none of them is returned. In the end there is no solution, and the planner
shows the closest schedulable plan.

| Number | Meaning | Source |
| --- | --- | --- |
| 280 | Plans the NN accepted but the CSP rejected | `python run_demo.py --profile tight`, or the web app Tight preset (NN yes, CSP no tile) |
| (4, 4, 3) | Closest schedulable plan, marks 62.7, 65.1, 65.4 | Same run |
| 21 of 30, mean 0.6 sessions | Real test profiles that got a plan; 9 were reported unreachable | `python -m eval.run_profiles --real`, `results/profiles_summary.csv` |

**Q: Why is the mean only 0.6 sessions?** Many real students are already predicted above their
generated target with no extra sessions, so their plan is (0, 0, 0).

**Q: How is the "closest plan" chosen?** Among the schedulable plans the search expanded, the one
whose worst assignment is closest to the target (largest min of mark minus target).

## Slide 14. Conclusions (9:15 to 10:00)

**Say:** One sentence each. The NN turns a plan into marks and is monotone, so more study never
hurts. A* finds the fewest sessions, as optimal as UCS with about a quarter of the nodes (34.5 vs
124.9). The CSP proves the plan fits the week, and AC-3 removes backtracking when slots are tight.
Next steps: more courses, real student timetables, and uncertainty on each mark. Then hand over to
the web app.

## Slides 15 to 18. The web app

### Slide 15. Overview

Start it with `python -m webapp` from the repo root. It opens `http://localhost:8000`.

- `webapp/server.py` is Python's built-in `http.server`, no extra packages.
- Endpoints: `GET /api/presets` (the three demo students), `POST /api/plan` (one run),
  `POST /api/compare` (all six method combinations).
- `/api/plan` calls `integration/planner.py` `plan_for`, the same function `run_demo.py` and the
  experiments use. The browser only draws; every number is computed on the server. So the app shows
  the same numbers as the report.

### Slide 16. Inputs

| Input | Where it goes |
| --- | --- |
| Preset | Example, Tight week, Target 100 are the three `run_demo.py` students. Custom keeps your values. |
| Target mark | `profile.target`, used in the goal test: every predicted mark must reach it |
| Free slots per week | CSP domain: slots 1 to N in every week |
| Average of past marks | NN features "previous mark" and "running average" for every assignment |
| Clicks per session | Clicks one planned session adds to the NN input (50 for demo students, 10 for real profiles) |
| Type, weight, due week | NN features; the due week also ends that assignment's CSP domain |
| Search, CSP, Marks | UCS / Greedy / A*; BT / BT+FC / +AC-3; monotone NN / average predictor |

How a session becomes NN input (`nn/effort.py`): one session adds "clicks per session" clicks to
that assignment's 4-week window before its deadline, split 50% content, 20% quizzes, 30% other,
and spread 5%, 15%, 30%, 50% over weeks 1 to 4 before the deadline. Assignments are sorted by due
week and named A1, A2, A3.

### Slide 17. The five numbers (the most likely TA questions)

| Tile | Exact meaning | Code | Example | Tight |
| --- | --- | --- | --- | --- |
| Sessions | `sum(plan)`. With no solution, the sum of the closest plan | `webapp/server.py` `run_plan` | 4 | 11 |
| Nodes expanded | Plans popped from the frontier and goal-tested | `search/adapter.py`, `SearchResult.nodes_expanded` | 5 (A*) | 1,331 |
| Plans checked by CSP | Distinct plans sent to the CSP (cached, so each once) | `integration/planner.py`, `len(out.csp_calls)` | 5 | 341 |
| NN yes, CSP no | Plans with every mark at or above target that did not fit | `out.nn_yes_csp_no` | 0 | 280 |
| Time | Wall-clock seconds for search plus CSP on the server | `out.seconds` | about 0.04 s | about 0.3 s |

Notes:
- 1,331 = 11 x 11 x 11: with no solution, search must try every plan before it can say no.
- With UCS the Example needs 22 nodes instead of 5; Greedy also needs 5.
- The CSP is called only when the NN says every mark reaches the target, or when the plan's worst
  margin beats the best fallback found so far. That is why "Plans checked" (341) is larger than
  "NN yes, CSP no" (280) on Tight: the extra 61 are fallback candidates or plans that passed.
- Time changes a little on every run; it is not a result, just a measurement.

### Slide 18. Charts

- **Predicted marks:** the server calls the predictor twice: with the plan (coloured bar) and with
  (0, 0, 0) (dark bar). The dashed line is the target. Example: A1 47.4 to 56.1, A2 53.9 to 56.2,
  A3 stays 56.7.
- **Timetable:** the CSP's answer, a dictionary (assignment, k) to (week, slot), drawn with weeks
  across and slots down, up to the last deadline. Red week labels are deadlines. Example: A1 in
  week 1, slots 1 to 3; A2 in week 2.
- **What if:** for each assignment and each n from 0 to 10, the NN mark with n sessions on that
  assignment and 0 on the others. Every line only goes up, which is the monotone design. A1 crosses
  55 at 3 sessions, A2 at 1, A3 at 0, which is exactly the plan (3, 1, 0). The ringed point is the
  chosen plan.

---

# Part 2: Abdo

## Slide 4. Data: OULAD course BBB (1:30 to 3:30)

**Say:** One row per student and assessment in course BBB. We train on three presentations and test
on a later one the model never saw, so the test is honest. Free slots and targets are not in OULAD,
so we generate them with a fixed seed. We left out personal attributes on purpose.

| Number | Meaning | Source |
| --- | --- | --- |
| 35,601 | Training rows, presentations 2013B, 2013J, 2014B | `python -m data.make_data`, `data/processed/train_data.csv` |
| 7,399 | Test rows, presentation 2014J | `data/processed/test_data.csv` |
| 18 | Features: previous mark, running average, weight, deadline week, type (TMA/CMA one-hot), and clicks in 3 groups (content, quizzes, other) for each of the 4 weeks before the deadline (3 x 4 = 12) | `data/schema.py`, `docs/data_dictionary.md` |
| 10.6 million | Daily click records in OULAD `studentVle.csv` | Raw data |
| {2, 3, 4} slots, target 50 to 70, 10 clicks per session | Generated values for the real profiles (seeded) | `data/profiles.py` |

Excluded: gender, age, region, disability and deprivation (IMD band).

**Q: Why split by presentation and not randomly?** A random split would put the same students and
the same cohort in both sets and make the test look better than it is. A later cohort is what the
planner would face in practice.

## Slide 5. Neural network (around 3:30)

**Say:** The network has two branches (`nn/model.py`). A base MLP reads the non-click features and
gives a starting mark. The effort branch reads log(1 + clicks) through weights forced to be positive
(softplus), so more clicks can never lower the mark. Trained with MSE loss and Adam, early stopping
on a validation split from the training years, seed 42 (`python -m nn.train --seed 42`). Weights
are in `nn/weights/`.

| Number | Meaning | Source |
| --- | --- | --- |
| 29.8 | Test RMSE on 2014J (MAE 22.2) | `nn/weights/baseline_comparison.txt`, `docs/results.md` |
| 28.9 | Plain MLP test RMSE (kept for comparison) | `nn/weights/mlp/baseline_comparison.txt` |
| 30.1 | Baseline: predict the training mean for everyone | Same file |
| 38.4 | Baseline: predict the student's previous average | Same file |
| 13.7 | Validation RMSE for both models | `nn/weights/train_log.json` |

**Explain:** Validation RMSE is 13.7, test is 29.8. The jump comes from the 2014J cohort, which has
a different set of assessments and lower marks, not from overfitting (validation is fine).

**Q: The plain MLP has lower RMSE. Why not use it?** See the next slide: it is slightly more
accurate but it says more study lowers the mark on 39% of steps, which makes it useless for planning.

## Slide 6. Why monotone (ends at 3:30)

**Say:** We found this when we connected the NN to search. With the plain MLP, adding a session
lowered the predicted mark on 39% of steps. The monotone network has 0% by construction.

| Number | Meaning | Source |
| --- | --- | --- |
| 39% to 0% | Steps (n to n+1 sessions) where the mark went down: MLP vs monotone | `python -m eval.monotonicity --real`, `results/monotonicity_summary.csv` |
| 57% vs 0% | Curves (0 to 10 sessions) that drop somewhere | Same |
| +1.0 to +7.7 | Average marks gained from 0 to 10 sessions: MLP vs monotone | Same |
| 2.68 | Largest gain one session gives on the test set; used by the A* heuristic | Same run |

## Slide 7. Search (3:30 to 5:00)

**Say:** (`search/`, entry point `search/adapter.py`)

- **State:** the plan (s1, s2, s3), sessions per assignment. Start (0, 0, 0).
- **Action:** add one session to one assignment. Cost 1, so path cost is total sessions and the
  cheapest goal is the fewest sessions.
- **Goal test:** every predicted mark is at least the target AND the CSP can schedule the plan.
- **UCS** orders by g (cost so far): optimal but wide. **Greedy** orders by h only: fast, no
  guarantee. **A\*** orders by g + h: optimal with an admissible h.
- **Heuristic:** for each assignment, take the marks still missing, divide by the biggest gain
  one session can give this student (computed from the NN before search) and round up. Add these
  up over the assignments. Example: an assignment 8 marks short with a maximum gain of 2.68 needs
  at least 3 more sessions.
- At most 10 sessions per assignment.

**Q: Why is the heuristic admissible?** No session can add more than the maximum gain, so each
assignment needs at least deficit / max gain sessions. A session only helps the assignment it is
given to, so these minimums add up. The sum never overestimates, so A* returns the optimal plan.

**Q: Is it consistent?** One step adds a session to one assignment and lowers that assignment's
deficit by at most the maximum gain, so h drops by at most 1 per step, which equals the step cost.

## Slide 8. Experiment 1: search (ends at 5:00)

**Setup:** 30 real test profiles, 5 seeds, targets 60, 70 and 80, 3 slots a week. Command:
`python -m eval.exp1_planner --real`. Output: `results/exp1_planner_summary.csv`, plots
`results/exp1_planner_*.png`.

| Number | Meaning |
| --- | --- |
| 124.9 / 34.5 / 8.9 | Nodes expanded on solved profiles at target 80: UCS / A* / Greedy |
| 0 | Extra sessions over UCS for A* and Greedy at every target (Greedy has no guarantee, it just matched here) |
| 71%, 60%, 25% | Profiles solved at targets 60, 70, 80 |
| 23.8 / 3.4 / 7.2 | Nodes at target 60: UCS / Greedy / A* |
| 1,331 | Nodes on unsolved profiles: all plans tried |

**Say:** A* is about 3.6 times cheaper than UCS and gives the same plan. The web app's "Nodes
expanded" tile is this same counter.

## Slide 12. Experiment 3: NN vs average (7:30 to 8:30)

**Setup:** the same planner with three predictors: the average of past marks (ignores sessions),
the plain MLP, the monotone NN. Command: `python -m eval.exp3_planner --real`
(`results/exp3_planner_summary.csv`) and `python -m eval.exp3_nn_vs_avg` for RMSE.

| Number | Meaning |
| --- | --- |
| 0% / 0% / 25% | Profiles solved at target 80: average / plain MLP / monotone NN |
| 74% (MLP 68%, average 74%) | Correct "already on track with 0 sessions or not" calls at target 60 |
| 29.8 vs 38.4 | Test RMSE: monotone NN vs previous average |

**Say:** The average predictor gives a fixed mark, so a plan can never raise it. The plain MLP barely
reacts to sessions. Only the monotone NN lets the planner reach high targets. The web app's Compare
methods tab runs this comparison live for one student.

## Slide 13. Limits (8:30 to 9:15)

Be direct:
- **Accuracy:** RMSE near 30 marks is large; 2014J differs from the training years.
- **Sessions are assumed:** one session = a fixed number of clicks is our choice, not measured.
  Free slots and targets are generated.
- **Correlation, not cause:** students who click more score more, but that does not prove clicking
  causes the marks. Monotonicity is a design choice, not a finding.
- **One course:** BBB only; other courses need retraining.

## Slide 19. Web app detail tabs

- **CSP solvers:** Experiment 2 on one plan. If there are rejected plans it uses the first one, the
  interesting case. Tight, plan (6, 4, 3): BT 15 assignments and 15 backtracks, FC 15 and 15 with
  113 values pruned, AC-3 0 and 0 (proves it impossible before search). The counting check is off
  here so all three solvers do the work.
- **Compare methods:** Experiments 1 and 3 on one student, 6 runs. Example: the NN finds (3, 1, 0)
  every time (nodes UCS 22, Greedy 5, A* 5); the average predictor gives 30, 30, 30 and finds
  nothing, because sessions do not change it.
- **Search trace:** for each expanded plan, the lowest (mark minus target). A plan can pass only
  when it reaches 0.
- **Rejected plans:** the NN yes, CSP no list, up to 20 shown. Tight: (6, 4, 3), (7, 4, 3),
  (6, 5, 3) and more.

---

# Part 3: Both. Live demo (5 minutes)

Before the talk: `python -m webapp` on the laptop, browser open on `http://localhost:8000`.
Backup if the laptop fails: `python run_demo.py` or the saved output in `docs/demo_backup/`.

| Run | Who | Settings | What to show |
| --- | --- | --- | --- |
| 1 | Zeyad | Example, A*, +AC-3, Neural net | Plan (3, 1, 0), 4 sessions, 5 nodes. Point at the timetable and the what-if lines crossing 55. |
| 2 | Zeyad | Tight week | No solution. 280 rejected plans, closest (4, 4, 3). Open Rejected plans and CSP solvers. |
| 3 | Abdo | Target 100 | No solution, 1,331 nodes: even 10 sessions stay below 100. Closest plan depends on the search: UCS (10, 6, 4), Greedy (10, 9, 5), A* (10, 7, 5). |
| 4 | Abdo | Example, Compare methods tab, Run all six | The NN plans every time; the average predictor cannot. |

**Q: Why does the closest plan for Target 100 differ between searches?** Each search keeps the best
plan among the ones it happened to expand, and they expand in different orders. None reaches 100.

---

# How to reproduce every number

```bash
pip install -r requirements.txt
python -m webapp                          # every web app number, live
python run_demo.py                        # Example; add --profile tight or --profile impossible
python -m data.make_data                  # needs OULAD in data/raw/: rebuilds the 35,601 / 7,399 rows
python -m nn.train --seed 42              # NN RMSE and baselines -> nn/weights/
python -m eval.monotonicity --real        # 39% / 0%, +1.0 / +7.7, 2.68
python -m eval.run_profiles --real        # 21 of 30 profiles, mean 0.6 sessions
python -m eval.exp1_planner --real        # Experiment 1
python -m eval.exp2_csp --real            # Experiment 2
python -m eval.exp3_planner --real        # Experiment 3
pytest -q                                 # all tests
```

All real-data numbers are also written down in `docs/results.md`.
