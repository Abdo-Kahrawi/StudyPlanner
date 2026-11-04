# Working together

| Role | Name | Branch prefix |
| --- | --- | --- |
| Abdo (data, nn, search) | TODO | `a/` |
| Zeyad (csp, integration, eval, demo) | TODO | `b/` |

## Rules

1. `main` is always runnable. Never push to it directly.
2. Work on short branches: `a/<topic>` or `b/<topic>`, for example `a/ucs-greedy`, `b/forward-checking`.
3. Every pull request is reviewed and approved by the other person. CI must be green.
4. Commit your own work under your own name and email, so `git shortlog -sn` shows contributions honestly.
5. Commit messages start with the module: `csp: add AC-3 with revise counter`.
6. Changes to `common/types.py` or `data/schema.py` need both people on the PR.
7. Raw and processed data are never committed; `python -m data.make_data` rebuilds them.
8. Weekly 30-minute sync at the start of each week to check the milestone.

## Milestones

| Tag | Week | Meaning |
| --- | --- | --- |
| M0 | 1 | Contract frozen, course chosen, stubs run |
| M1 | 3 | Real features + NN v1; CSP passes unit tests |
| M2 | 4 | End-to-end plan on one profile (stubs allowed) |
| `m3-minimum` | 5 | Minimum version on 30 profiles |
| M4 | 6 | All three experiment tables and plots |
| `v1.0-submission` | 7 | Code freeze |
