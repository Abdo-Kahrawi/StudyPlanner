"""make_data.py - build the NN training data from raw OULAD. Owner: Abdo.

Originally exported from the Colab notebook make_data.ipynb (kept at the repo root for the
exploration and plots). This module keeps the cleaning rules from the notebook, without the
exploratory prints, so that it can be run from the repo root:

    python -m data.make_data                  # download OULAD into data/raw/ if missing, then build
    python -m data.make_data --no-download    # raw CSVs already in data/raw/

Steps:
    1. download_oulad(): OULAD CSVs -> data/raw/
    2. clean_raw():      data/raw/*.csv -> data/interim/*_c.csv  (rules in the dictionary below)
    3. make_dataset():   data/interim/ -> data/processed/train_data.csv, test_data.csv
"""
import argparse
import csv
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import config

# The OU download link now redirects to a web page; UCI hosts the same zip.
OULAD_URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
RAW_FILES = ['assessments.csv', 'courses.csv', 'studentAssessment.csv',
             'studentRegistration.csv', 'vle.csv', 'studentVle.csv']
INTERIM_DIR = config.ROOT / 'data' / 'interim'

"""
===============================================================================
                       DATA DICTIONARY & PIPELINE DOCSTRING
===============================================================================

Project: Assessment-Aware Study Planner (P1)
Dataset Source: Open University Learning Analytics Dataset (OULAD)
Description:
    This module cleans, transforms, and aggregates raw OULAD CSV files into
    reproducible training and testing datasets for neural network score
    prediction and CSP study timetable generation.

-------------------------------------------------------------------------------
1. FILE SUMMARY & PRIMARY KEYS
-------------------------------------------------------------------------------
- courses.csv
    Keys: ['code_module', 'code_presentation']
    Description: Metadata regarding course presentations and overall module lengths.

- ass_c.csv (Cleaned from assessments.csv)
    Keys: ['id_assessment']
    Description: Assessment metadata including deadlines, types, and relative weights.

- student_assessment_c.csv (Cleaned from studentAssessment.csv)
    Keys: ['id_assessment', 'id_student']
    Description: Records of student submission times, scores, and calculated metrics.

- student_registration_c.csv (Cleaned from studentRegistration.csv)
    Keys: ['code_module', 'code_presentation', 'id_student']
    Description: Enrolment timestamps and withdrawal statuses per student presentation.

- vle_c.csv (Cleaned from vle.csv)
    Keys: ['id_site']
    Description: VLE resource catalog filtered to active study materials.

- student_vle_c.csv (Cleaned from studentVle.csv)
    Keys: ['id_student', 'id_site', 'date']
    Description: Granular clickstream activity tracking daily user-resource interactions.

- studentInfo.csv
    Keys: ['id_student', 'code_module', 'code_presentation']
    Description: Student demographic data (EXCLUDED from model inputs per ethical rules).

-------------------------------------------------------------------------------
2. TABLE SCHEMAS & PREPROCESSING RULES
-------------------------------------------------------------------------------
[Table: ass_c.csv]
- code_module (str)           : Unique identifier for the academic module (Real).
- code_presentation (str)     : Semester presentation identifier (Real).
- id_assessment (int)         : Unique identifier for the assessment (Real).
- assessment_type (str)       : Category of evaluation: 'TMA', 'CMA', or 'Exam' (Real).
- date (float)                : Deadline day relative to course start (Imputed).
                                Rule: Missing 'Exam' dates are imputed using the
                                corresponding 'module_presentation_length' from courses.csv.
- weight (float)              : Percentage contribution toward module completion (Real).

[Table: student_assessment_c.csv]
- id_assessment (int)         : Identifier connecting to ass_c.csv (Real).
- id_student (int)            : Unique student identifier (Real).
- date_submitted (int)        : Day the student submitted their work (Real).
- is_banked (int)             : Flag indicating result carried over from past term (Real).
- score (float)               : Score awarded from 0 to 100 (Cleaned).
                                Rule: Missing scores on non-withdrawn attempts filled with 0.
- Difference (int)            : Calculated relative delay = date_submitted - date (Engineered).
- Submission Category (str)   : 'Early' if Difference < 0 else 'Late' (Engineered).
- assessment_mean (float)     : Mean cohort score for the given id_assessment (Engineered).
- performance_label (str)     : Categorical performance banding (Engineered).
                                Categories: 'Terrible' (<60), 'Low' (<mean - 10),
                                'Acceptable' (mean ± 10), 'Good' (>mean + 10).

[Table: student_registration_c.csv]
- code_module (str)           : Unique module identifier (Real).
- code_presentation (str)     : Module presentation identifier (Real).
- id_student (int)            : Unique student identifier (Real).
- date_registration (float)   : Day student registered relative to start (Cleaned).
                                Rule: Missing values defaulted to 0.
- date_unregistration (float) : Day student withdrew, or NaN if completed (Real).
- Category (str)              : 'Withdrawn' if date_unregistration exists else 'Not Withdrawn' (Engineered).

[Table: vle_c.csv]
- id_site (int)               : Unique VLE material identifier (Real).
- code_module (str)           : Module code (Real).
- code_presentation (str)     : Presentation code (Real).
- activity_type (str)         : Filtered active study types: ['oucontent', 'quiz',
                                'resource', 'url', 'externalquiz', 'glossary', 'dataplus'] (Real).
- week_from (float)           : Starting week material is available (Imputed; NaN -> 1).
- week_to (float)             : Ending week material is available (Imputed; NaN -> max_weeks).

[Table: student_vle_c.csv]
- code_module (str)           : Module code with quote artifacts trimmed (Cleaned).
- code_presentation (str)     : Presentation code with quote artifacts trimmed (Cleaned).
- id_student (int)            : Unique student identifier (Real).
- id_site (int)               : Target VLE site identifier (Cleaned; 'nan' strings dropped).
- date (int)                  : Day offset relative to start (negative = pre-start clicks) (Real).
- sum_click (int)             : Total clicks recorded on site for that day (Real).

-------------------------------------------------------------------------------
3. ENGINEEREED NEURAL NETWORK INPUT MATRIX (~22 COLUMNS)
-------------------------------------------------------------------------------
The final train_data.csv and test_data.csv matrices produced for course BBB/FFF
contain the following feature layout:

- Metadata / Keys : 'id_student', 'id_assessment', 'code_presentation'
- Target Variable : 'target_score' (Continuous 0–100 assessment score)
- Past Metrics    : 'prev_score' (Shifted score of immediately preceding assignment),
                    'cumulative_avg_score' (Expanding mean of past assignment scores)
- Course Context  : 'weight', 'weeks_since_start'
- Clicks (12 cols): Clicks aggregated across 3 activity groups ('group_content',
                    'group_quizzes', 'group_other') over 4 pre-deadline weeks:
                    Week 1: [deadline - 28, deadline - 21)
                    Week 2: [deadline - 21, deadline - 14)
                    Week 3: [deadline - 14, deadline - 7)
                    Week 4: [deadline - 7, deadline]
- One-Hot Categoricals: 'assessment_type_TMA', 'assessment_type_CMA', 'assessment_type_Exam'

-------------------------------------------------------------------------------
4. GENERATED PLANNER SIMULATION PARAMETERS
-------------------------------------------------------------------------------
Deterministic rules (fixed seed=42) used during Search + CSP planning:
- Session Conversion Rate : 1 Study Session = 50 VLE clicks.
- Weekly Slot Capacity    : Maximum 3 free study slots per calendar week.
- Target Mark             : Specified target score per student assessment (e.g., 55/100).

-------------------------------------------------------------------------------
5. ETHICAL COMPLIANCE & EXCLUSIONS
-------------------------------------------------------------------------------
- Demographic Exclusion: Fields from studentInfo.csv ('gender', 'region', 'highest_education',
  'imd_band', 'age_band', 'disability') are strictly excluded from neural network
  inputs to comply with fairness guidelines and eliminate bias.
- Noise Reduction: Records where students withdrew from the course and scored 0
  are excluded from non-zero target distributions to prevent score deflation.
===============================================================================
"""


def download_oulad(raw_dir=config.RAW_DIR, url=OULAD_URL):
    """Download and unzip the OULAD CSVs into raw_dir, unless they are already there."""
    raw_dir = Path(raw_dir)
    if all((raw_dir / f).exists() for f in RAW_FILES):
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading OULAD from {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
    except OSError as e:
        raise SystemExit(f"Download failed ({e}). Download the zip from {url} "
                         f"and unzip the CSVs into {raw_dir}")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in z.namelist():
            name = os.path.basename(member)
            if name.endswith('.csv'):
                (raw_dir / name).write_bytes(z.read(member))
    print(f"Saved OULAD CSVs to {raw_dir}")


def clean_raw(raw_dir=config.RAW_DIR, interim_dir=INTERIM_DIR):
    """Apply the cleaning rules from the data dictionary; write the *_c.csv tables."""
    raw_dir, interim_dir = Path(raw_dir), Path(interim_dir)
    missing = [f for f in RAW_FILES if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw OULAD files in {raw_dir}: {missing}")
    interim_dir.mkdir(parents=True, exist_ok=True)

    assessments = pd.read_csv(raw_dir / 'assessments.csv', na_values='?')
    courses = pd.read_csv(raw_dir / 'courses.csv', na_values='?')
    student_ass = pd.read_csv(raw_dir / 'studentAssessment.csv', na_values='?')
    student_reg = pd.read_csv(raw_dir / 'studentRegistration.csv', na_values='?')
    vle = pd.read_csv(raw_dir / 'vle.csv', na_values='?')

    # Assessments: missing Exam dates -> module_presentation_length
    ass = assessments.merge(courses[['code_module', 'code_presentation', 'module_presentation_length']],
                            on=['code_module', 'code_presentation'], how='left')
    ass['date'] = np.where((ass['assessment_type'] == 'Exam') & (ass['date'].isna()),
                           ass['module_presentation_length'], ass['date'])
    ass = ass.drop(columns=['module_presentation_length'])

    # Student assessments: drop rows without a student, Difference / Submission Category,
    # missing scores -> 0
    sa = ass.merge(student_ass, on=['id_assessment'], how='left')
    sa = sa.dropna(subset=['id_student'])
    sa['Difference'] = sa['date_submitted'] - sa['date']
    sa['Submission Category'] = np.where(sa['Difference'] < 0, 'Early', 'Late')
    sa['score'] = sa['score'].fillna(0)
    sa = sa[['id_assessment', 'id_student', 'date_submitted', 'Difference',
             'Submission Category', 'is_banked', 'score']].copy()
    sa['id_student'] = sa['id_student'].astype(int)

    # OULAD marks missing values with '?', read as NaN above.
    # Registrations: drop students with neither registration nor unregistration date,
    # missing registration date -> 0, Category = Withdrawn / Not Withdrawn
    reg = student_reg.copy()
    reg['Category'] = np.where(reg['date_unregistration'].isna(), 'Not Withdrawn', 'Withdrawn')
    no_dates = reg[reg['date_registration'].isna() & reg['date_unregistration'].isna()]['id_student']
    reg = reg[~reg['id_student'].isin(no_dates)].copy()
    reg['date_registration'] = reg['date_registration'].fillna(0)

    # Drop 0 scores of students who withdrew; cohort mean and performance label
    withdrawn = student_reg[student_reg['date_unregistration'].notna()]['id_student']
    sa = sa[~(sa['id_student'].isin(withdrawn) & (sa['score'] == 0))].copy()
    sa['assessment_mean'] = sa.groupby('id_assessment')['score'].transform('mean')
    sa['performance_label'] = np.select(
        [sa['score'] < 60, sa['score'] < sa['assessment_mean'] - 10,
         sa['score'] <= sa['assessment_mean'] + 10],
        ['Terrible', 'Low', 'Acceptable'], default='Good')

    # VLE: keep active study types, impute week_from (1) and week_to (course length in weeks)
    active_types = ['oucontent', 'quiz', 'resource', 'url', 'externalquiz', 'glossary', 'dataplus']
    v = vle[vle['activity_type'].isin(active_types)].copy()
    v = v.merge(courses[['code_module', 'code_presentation', 'module_presentation_length']],
                on=['code_module', 'code_presentation'], how='left')
    v['week_from'] = v['week_from'].fillna(1)
    v['week_to'] = v['week_to'].fillna(np.ceil(v['module_presentation_length'] / 7))
    v = v.drop(columns=['module_presentation_length'])

    # Student VLE: read without quote parsing (parse error at line 1133148), strip quotes,
    # drop rows with no site
    svle = pd.read_csv(raw_dir / 'studentVle.csv', quoting=csv.QUOTE_NONE)
    svle.columns = svle.columns.str.replace('"', '').str.strip()
    for col in svle.select_dtypes(include=['object', 'string']).columns:
        svle[col] = svle[col].astype(str).str.replace('"', '').str.strip()
    svle = svle[~svle['id_site'].astype(str).isin(['nan', ''])]
    for col in ['id_student', 'id_site', 'date', 'sum_click']:
        svle[col] = pd.to_numeric(svle[col])

    out = {'ass_c.csv': ass, 'courses.csv': courses, 'student_assessment_c.csv': sa,
           'student_registration_c.csv': reg, 'vle_c.csv': v, 'student_vle_c.csv': svle}
    for name, df in out.items():
        df.to_csv(interim_dir / name, index=False)
        print(f"Saved: {interim_dir / name} {df.shape}")


#Finalizing our data and finishing off.
def make_dataset(
    course_code='BBB', #Filtering depending on one course, to focus our attention on one distribution, eliminating discrepencies with other courses' distributions
    train_presentations=['2013B', '2013J', '2014B'],
    test_presentations=['2014J'],
    seed=42,
    output_dir=None,
    interim_dir=None,
):
    """
    Generates dataset features and target labels for P1. Assessment-Aware Study Planner.
    Strictly excludes demographic information (gender, age, region, disability, imd_band).
    """
    np.random.seed(seed)
    output_dir = str(output_dir or config.PROCESSED_DIR)
    interim_dir = Path(interim_dir or INTERIM_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Data Pipeline for Course {course_code} (Seed: {seed}) ---")

    # 1. Load Raw CSV Files
    assessments_df = pd.read_csv(interim_dir / 'ass_c.csv')
    courses_df = pd.read_csv(interim_dir / 'courses.csv')
    student_ass_df = pd.read_csv(interim_dir / 'student_assessment_c.csv')
    vle_df = pd.read_csv(interim_dir / 'vle_c.csv')
    student_vle_df = pd.read_csv(interim_dir / 'student_vle_c.csv')
    #We're not going to use either the student_registeration.csv or studentInfo.csv, because student_info contains features incompatible with project requirements.
    #Student registration could be helpful in EDA, but not so much in generating our train/test data for our NN.

    # 2. Filter by Course Module
    assessments = assessments_df[assessments_df['code_module'] == course_code].copy() #Making copies,as to not alter the csvs themselves
    vle = vle_df[vle_df['code_module'] == course_code].copy()
    student_vle = student_vle_df[student_vle_df['code_module'] == course_code].copy()

    # Handle missing assessment dates (Exams default to module length)
    assessments = assessments.merge(
        courses_df[['code_module', 'code_presentation', 'module_presentation_length']],
        on=['code_module', 'code_presentation'],
        how='left' #Left Join
    )
    assessments['date'] = np.where(
        (assessments['assessment_type'] == 'Exam') & (assessments['date'].isna()),
        assessments['module_presentation_length'],
        assessments['date']
    )
    assessments.drop(columns=['module_presentation_length'], inplace=True) #Now that we have replaced the values we needed to replace,
    #We no longer need this column with us in the assessment thing

    # 3. Categorize VLE Activity into 3 Specific Groups 🏷️
    def categorize_activity(activity):
        if activity in ['oucontent', 'resource', 'page', 'url']:
            return 'group_content'
        elif activity in ['quiz', 'externalquiz', 'subpage']:
            return 'group_quizzes'
        else:
            return 'group_other'

    vle['activity_group'] = vle['activity_type'].apply(categorize_activity)
    student_vle = student_vle.merge(vle[['id_site', 'activity_group']], on='id_site', how='left')
    student_vle['activity_group'] = student_vle['activity_group'].fillna('group_other')

    # 4. Merge Student Assessment with Assessment Details
    df_merged = student_ass_df.merge(
        assessments, on='id_assessment', how='inner'
    )

    # Sort chronologically to compute cumulative previous assessment scores
    df_merged.sort_values(by=['id_student', 'code_presentation', 'date'], inplace=True)

    # Calculate previous score features
    df_merged['prev_score'] = df_merged.groupby(['id_student', 'code_presentation'])['score'].shift(1).fillna(0)
    df_merged['cumulative_avg_score'] = (
        df_merged.groupby(['id_student', 'code_presentation'])['score']
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0) #The core purpose of using shift here is that we eliminate target leakage. Meaning, if the NN is predicting assignment 2 score, it shouldn't KNOW the score of assignment 2. Thus, we shift them all one position backwards.
        #Score of assignment one goes away. Score of assignment 2 becomes assignment 1 score and so on.
    )

    # 3. FAST VECTORIZED JOIN: Clicks to Assessments 🚀
    print("Performing vectorized click aggregation across 4 pre-deadline weeks...")

    # Merge clicks with assessment deadlines matching on student & presentation
    clicks_merged = student_vle.merge(
        df_merged[['id_student', 'code_presentation', 'id_assessment', 'date']],
        on=['id_student', 'code_presentation'],
        how='inner'
    )
    # Keep only clicks in the 28 days before the assessment deadline
    # (vle date >= deadline - 28 AND vle date <= deadline)
    valid_clicks = clicks_merged[
        (clicks_merged['date_x'] >= clicks_merged['date_y'] - 28) &
        (clicks_merged['date_x'] <= clicks_merged['date_y'])
    ].copy()

    # Compute week number (1 to 4) relative to deadline
    # Week 1: [deadline - 28, deadline - 21), ..., Week 4: [deadline - 7, deadline]
    day_diff = valid_clicks['date_x'] - (valid_clicks['date_y'] - 28) #Calculating the difference in days to help the next column in labeling the week of the click on the VLE thing.
    valid_clicks['week_num'] = np.clip(np.ceil(day_diff / 7.0).astype(int), 1, 4) #Here, we're assigning a number from 1 to 4 to each VLE, as in when it was clicked relative to its deadline.

    # Pivot table to aggregate click sums per (id_student, id_assessment)
    pivot_clicks = valid_clicks.pivot_table(
        index=['id_student', 'id_assessment'],
        columns=['activity_group', 'week_num'],
        values='sum_click',
        aggfunc='sum',
        fill_value=0
    )

    # Flatten multi-level columns (e.g. ('group_content', 1) -> 'group_content_week1')
    pivot_clicks.columns = [f"{grp}_week{wk}" for grp, wk in pivot_clicks.columns]
    pivot_clicks.reset_index(inplace=True)

    # 4. Merge Pivoted Click Features back to main assessment table
    full_dataset = df_merged.merge(pivot_clicks, on=['id_student', 'id_assessment'], how='left')

    # Ensure missing click combinations are 0
    expected_click_cols = [f"{grp}_week{wk}" for grp in ['group_content', 'group_quizzes', 'group_other'] for wk in range(1, 5)]
    for col in expected_click_cols:
        if col not in full_dataset.columns:
            full_dataset[col] = 0
        else:
            full_dataset[col] = full_dataset[col].fillna(0)

    # Add time and meta features
    full_dataset['weeks_since_start'] = np.ceil(full_dataset['date'] / 7.0)
    full_dataset.rename(columns={'score': 'target_score'}, inplace=True)

    # One-hot encode assessment_type
    full_dataset = pd.get_dummies(full_dataset, columns=['assessment_type'], drop_first=False)

    # Select output columns
    feature_cols = (
        ['id_student', 'id_assessment', 'code_presentation', 'target_score',
         'prev_score', 'cumulative_avg_score', 'weight', 'weeks_since_start'] +
        expected_click_cols +
        [c for c in full_dataset.columns if c.startswith('assessment_type_')]
    )
    final_df = full_dataset[feature_cols].copy()

    # 5. Chronological Train / Test Split
    train_df = final_df[final_df['code_presentation'].isin(train_presentations)].copy()
    test_df = final_df[final_df['code_presentation'].isin(test_presentations)].copy()

    # Save outputs
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"✅ Fast preprocessing complete!")
    print(f"   Train set: {train_df.shape} -> saved to {train_path}")
    print(f"   Test set:  {test_df.shape} -> saved to {test_path}")
    # 8. Output Data Dictionary Summary
    print("\n--- Data Dictionary Summary ---")
    dict_summary = """
    Target Column:
    - target_score: Real score earned by student on the assessment (0 - 100).

    Input Features (12 core feature categories):
    1. group_content_week[1-4]: Clicks on core content resources across 4 weeks pre-deadline.
    2. group_quizzes_week[1-4]: Clicks on practice quizzes & subpages across 4 weeks pre-deadline.
    3. group_other_week[1-4]: Clicks on forums, homepages, etc. across 4 weeks pre-deadline.
    4. prev_score: Score on immediately preceding assessment.
    5. cumulative_avg_score: Mean score on all previous course assessments.
    6. weight: Official assessment weight percentage.
    7. weeks_since_start: Number of weeks from course start to assessment deadline.
    8. assessment_type_*: One-hot encoded indicators for assessment types (TMA, CMA, Exam).

    * Note: Demographic features (gender, age, region, disability, imd_band) strictly excluded.
    """
    print(dict_summary)
    return train_df, test_df



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--raw_dir', default=str(config.RAW_DIR))
    ap.add_argument('--out_dir', default=str(config.PROCESSED_DIR))
    ap.add_argument('--course', default='BBB')
    ap.add_argument('--seed', type=int, default=config.SEED)
    ap.add_argument('--no-download', action='store_true')
    args = ap.parse_args(argv)
    if not args.no_download:
        download_oulad(args.raw_dir)
    clean_raw(args.raw_dir, INTERIM_DIR)
    make_dataset(course_code=args.course, seed=args.seed, output_dir=args.out_dir,
                 interim_dir=INTERIM_DIR)


if __name__ == '__main__':
    main()
