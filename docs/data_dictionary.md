# Data dictionary

Owner: Person A (generated profile fields: Person B). Draft Week 3, final Week 7.

| Column | Source table | Real or generated | Meaning / how generated |
| --- | --- | --- | --- |
| id_student | studentInfo | real | Student id |
| score | studentAssessment | real | Target, 0-100 |
| ... | | | Fill in every feature in `data/schema.py:FEATURES` |
| slots_per_week | — | generated | Rule: TODO |
| target | — | generated | Rule: TODO |
| clicks_per_session | — | generated | Rule: TODO |

Excluded on purpose: gender, age_band, region, disability, imd_band.
