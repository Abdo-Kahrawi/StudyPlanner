# Data dictionary

Owner: Person A (generated profile fields: Person B). Draft Week 3, final Week 7.

| Column | Source table | Real or generated | Meaning / how generated |
| --- | --- | --- | --- |
| id_student | studentInfo | real | Student id |
| score | studentAssessment | real | Target, 0-100 |
| ... | | | Fill in every feature in `data/schema.py:FEATURES` |
| slots_per_week | — | generated | Uniform choice from {2, 3, 4}, seeded (`data/profiles.py`) |
| target | — | generated | Uniform choice from {50, 55, 60, 65, 70}, seeded |
| clicks_per_session | — | generated | One third of a typical week of activity: median over training rows of (sum of the three 4-week click features) / 4 / 3, rounded, minimum 10; 50 if no data |
| start_week | — | generated | 0: planning starts at the beginning of the presentation |
| deadline_week | assessments.date via `weeks_since_start` | real | Assessment deadline in weeks from presentation start, rounded up |

Excluded on purpose: gender, age_band, region, disability, imd_band.
