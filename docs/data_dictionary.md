# Data dictionary

Owner: Abdo (generated profile fields: Zeyad).
Built by `python -m data.make_data` (course **BBB**, train presentations 2013B, 2013J, 2014B,
test presentation 2014J, seed 42). Column names are fixed in `data/schema.py`.

## Raw tables used (OULAD, CC BY 4.0)

| File | Used for |
| --- | --- |
| assessments.csv | assessment type, deadline (`date`, days from start), weight |
| courses.csv | module length (imputes missing Exam dates) |
| studentAssessment.csv | scores (target and previous-score features) |
| studentRegistration.csv | withdrawals (0 scores of withdrawn students are dropped) |
| vle.csv | activity type of each VLE page (activity groups) |
| studentVle.csv | daily clicks per student and page |
| studentInfo.csv | **not used**: demographic fields are excluded on purpose |

## Cleaning rules (`clean_raw`, output in `data/interim/`)

- Missing Exam deadlines -> module length in days.
- Student-assessment rows without a student are dropped; missing scores of submitted work -> 0.
- 0 scores of students who withdrew are dropped (withdrawal, not performance).
- Registrations with neither registration nor unregistration date are dropped; missing
  registration date -> 0.
- VLE pages kept: oucontent, quiz, resource, url, externalquiz, glossary, dataplus. Missing
  `week_from` -> 1, missing `week_to` -> module length in weeks.
- studentVle is read without quote parsing (a malformed quote at line 1 133 148), quotes are
  stripped and rows without a page are dropped.

## NN table (`data/processed/train_data.csv`, `test_data.csv`): one row per (student, assessment)

| Column | Source | Real or generated | Meaning |
| --- | --- | --- | --- |
| id_student, id_assessment, code_presentation | studentAssessment, assessments | real | keys |
| target_score | studentAssessment.score | real | NN target, 0-100 |
| prev_score | studentAssessment | real | score on the student's previous assessment in this presentation, 0 if none |
| cumulative_avg_score | studentAssessment | real | mean of all the student's earlier scores in this presentation, 0 if none |
| weight | assessments | real | assessment weight (%) |
| weeks_since_start | assessments.date | real | deadline week, `ceil(date / 7)` |
| group_content_week1..4 | studentVle + vle | real | clicks on oucontent, resource, page, url |
| group_quizzes_week1..4 | studentVle + vle | real | clicks on quiz, externalquiz, subpage |
| group_other_week1..4 | studentVle + vle | real | all other clicks |
| assessment_type_CMA, assessment_type_TMA | assessments | real | one-hot assessment type |

Click weeks count back from the deadline: week 1 = days [deadline-28, deadline-21),
week 2 = [-21, -14), week 3 = [-14, -7), week 4 = [-7, deadline]. Previous scores are shifted by
one assessment so the NN never sees the score it predicts.

Split: by presentation (train = earlier presentations, test = 2014J), so the test set is a later
cohort, not a random sample.

## Planner profiles (`data/profiles.py`)

| Field | Source | Real or generated | Meaning / how generated |
| --- | --- | --- | --- |
| student_id | test rows | real | sampled with the seed among 2014J students with 3+ TMAs |
| assignments | test rows | real | the student's first 3 TMAs: id, weight, `deadline_week = ceil(weeks_since_start)` |
| history `<id>/<feature>` | test rows | real | each assignment's prev_score, cumulative_avg_score and 12 click features (NN inputs before any planned session) |
| history `prev_score_mean` | test rows | real | the student's average before the first planned TMA |
| slots_per_week | — | generated | uniform choice from {2, 3, 4}, seeded |
| target | — | generated | uniform choice from {50, 55, 60, 65, 70}, seeded |
| clicks_per_session | train rows | generated | one third of a typical week of activity: median over training rows of (sum of the 12 click features) / 4 / 3, rounded, minimum 10 |
| start_week | — | generated | 0: planning starts at the beginning of the presentation |

## Effort rule (`nn/effort.py`)

One planned session for an assignment adds `clicks_per_session` clicks to that assignment's
4-week window: 50% content, 20% quizzes, 30% other, spread 5% / 15% / 30% / 50% over weeks 1-4
(more recent weeks get more). All other features stay the same. This is a modelling assumption,
not something measured in the data.

Excluded on purpose: gender, age_band, region, highest_education, disability, imd_band.
