# Report draft: scheduling side (Person B)

Draft sections for the final report, covering `csp/`, `integration/`, `data/profiles.py`,
`eval/harness.py`, Experiment 2 and the planner tests. Numbers marked *(stub)* were produced
with `common/stubs.py` in place of the NN and search, and must be re-run once Person A's
modules are merged (`python -m eval.exp2_csp --real`).

## 1. Scheduling as a CSP

Given a plan `(n_1, n_2, n_3)` (sessions per assignment), the CSP places every session in a
weekly study slot.

- **Variables:** one per session, `(assignment_id, k)` for `k = 0 .. n_a - 1`.
- **Domains:** slot indices `t = week * slots_per_week + slot`, for
  `start_week <= week < deadline_week` of that assignment. A session must finish before its
  deadline week.
- **Constraints:**
  1. one session per slot: `t_x != t_y` for every pair of sessions (all-different);
  2. sessions of one assignment stay in order: `t_(a,k) < t_(a,j)` for `k < j`. This removes
     symmetric solutions that only swap identical sessions.
- Every pair of variables shares a constraint, so the constraint graph is complete.

Variables are tried in a fixed order (assignments by deadline, sessions in order) and values
from the earliest slot, so the three solver modes differ only in how much they prune.

## 2. Algorithms

| Mode | What it does |
| --- | --- |
| BT | Chronological backtracking; a value is tried only if it is consistent with the current assignment. |
| BT+FC | After each assignment, Forward Checking removes now-impossible values from the unassigned variables and backtracks at once if a domain empties. |
| BT+FC+AC-3 | AC-3 runs once before the search (all arcs), then BT+FC. With the order constraints, AC-3 trims each session's domain to the slots it can actually reach. |
| MAC (extension) | AC-3 after every assignment, seeded with the arcs into the variable just set. |

Counters reported for every run: `assignments`, `backtracks`, `values_pruned` (split into
`fc_removed` / `ac3_removed`) and `revisions`. A run stops after `max_assignments` and logs
that it hit the limit, so the worst case is bounded.

Inside the planner a cheap counting check runs first: for every deadline, the sessions due by
then must fit in the slots before it. This rejects most impossible plans without search. It is
switched **off** in Experiment 2, so every mode does its own work.

## 3. The planner loop (integration)

1. Search (UCS, Greedy or A*) proposes the next plan.
2. The NN predicts the mark of each assignment under that plan. Every mark must reach the target.
3. The CSP checks whether the plan fits the student's weekly slots. Results are cached per plan.
4. If both pass, the plan and its timetable are returned. If the NN passes but the CSP fails,
   the plan is recorded as an "NN yes, CSP no" event and the search continues.
5. If the search runs out of plans, the planner prints
   *"No solution: target unreachable at this capacity"* with the best predicted marks and the
   plan that reached them.

The demo (`run_demo.py`) shows all three outcomes: the worked example from the brief,
`--profile tight` (the NN says enough, the CSP rejects the schedule) and `--profile impossible`.

## 4. Student profiles

Real fields (history, clicks, assignments and deadlines) come from the processed OULAD feature
table. Generated fields use a fixed seed and are documented in `docs/data_dictionary.md`:
`slots_per_week` from {2, 3, 4}, `target` from {50, 55, 60, 65, 70},
`clicks_per_session` = one third of a typical week of activity (median, at least 10), and
`start_week = 0`.

## 5. Experiment 2: BT vs BT+FC vs BT+FC+AC-3

**Setup.** For each of 5 seeds, 30 profiles. The planner picks a plan at 3 slots per week; that
same plan is then scheduled by each mode at 2, 3 and 4 slots per week. Fewer slots make the same
plan tighter. Limit: 100 000 assignments. Values are the mean (std) over seeds, after first
averaging within each seed.

**Results** *(stub, synthetic profiles, 2026-11-05)*:

| Slots/week | Mode | Assignments | Backtracks | Values pruned | Seconds | Solved | Hit limit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | BT | 886 (1812) | 877 (1811) | 0 | 0.063 | 77% | 0.8% |
| 2 | BT+FC | 886 (1812) | 877 (1811) | 7936 | 0.012 | 77% | 0.8% |
| 2 | BT+FC+AC-3 | 236 (508) | 227 (508) | 2209 | 0.020 | 77% | 0% |
| 3 | all modes | 15.2 (2.5) | 0 | 0 / 156 / 203 | ≤ 0.05 | 100% | 0% |
| 4 | all modes | 15.2 (2.5) | 0 | 0 / 156 / 203 | ≤ 0.075 | 100% | 0% |

Plots: `results/exp2_assignments.png`, `results/exp2_backtracks.png`, `results/exp2_seconds.png`.

**Reading.**
- With 3 or 4 slots per week, every plan fits on the first try (one assignment per session, no
  backtracks). Pruning only costs time there; AC-3's preprocessing is the slowest mode.
- With 2 slots per week, the plans become tight and about a quarter cannot be scheduled.
  AC-3 cuts assignments and backtracks by about 3.7x and is the only mode that never hits the
  limit, because it finds failing plans before the search starts.
- FC makes the same choices as BT under this variable order, so its counts match, but it is
  about 5x faster: it detects dead ends from domains instead of re-checking the whole
  assignment for every value.
- The large std at 2 slots comes from a few hard plans that dominate the mean. The report
  should also show the median or a per-profile plot.

## 6. Tests

`pytest -q`: 38 passed, 2 skipped (skips are Person A tests that need real data).

- CSP: a tiny known-answer case, validity of every returned schedule, infeasible plans, counters
  showing the pruning differences, AC-3 order-bound trimming, MAC agreeing with BT+FC, and the
  counting check and limit.
- Planner: 5 normal cases (brief example, strong student, weak student, 2 and 4 slots per week),
  3 edge cases (target already met, deadline in the first week, exactly enough slots) and
  2 impossible cases (target 100, too few slots before a deadline).
- Profiles: reproducible synthetic profiles, and building profiles from a feature table.

## 7. Limitations (scheduling side)

- All sessions are the same length, and every slot fits one session.
- The fixed variable order keeps the comparison fair, but MRV or LCV would solve the tight cases faster.
- The complete constraint graph makes AC-3 quadratic in the number of sessions. That is fine at
  10 sessions per assignment, but it would need a dedicated all-different propagator at larger sizes.
- Stub numbers are for development only. The final tables must use the real NN and search.
