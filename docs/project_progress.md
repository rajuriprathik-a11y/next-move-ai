# Next Move AI — Project Progress

## Project Objective

Predict whether a student's next eligible main question in the ASSISTments 2009–2010 Skill Builder dataset will be recorded correct, using their previously observed interaction history.

## 1. Dataset Understanding and Audit

**Status: Completed**

- Loaded and examined the ASSISTments 2009–2010 Skill Builder dataset.
- Original dataset: 346,860 interactions and 31 columns.
- Identified main questions (`original = 1`) and scaffolding questions (`original = 0`).
- Selected main questions as prediction-eligible records.
- Examined missing values, target distribution, student IDs and interaction ordering.
- Verified that `order_id` provides a consistent interaction ordering within each student.

## 2. Data Cleaning and Preparation

**Status: Completed**

- Retained 275,458 main-question interactions.
- Parsed multiple skill IDs into Python sets.
- Represented missing skill IDs using empty sets.
- Sorted interactions by `user_id` and `order_id`.
- Selected the seven required columns:
  - `user_id`
  - `order_id`
  - `problem_id`
  - `correct`
  - `skill_id`
  - `skill_set`
  - `attempt_count`
- Saved the cleaned dataset to:

`data/processed/main_questions_cleaned.csv`

## 3. Student-Level Data Splitting

**Status: Completed**

- Split 4,217 unique students into an 80% training pool and 20% final test set.
- Training pool: 3,373 students and 219,983 interactions.
- Final test set: 844 students and 55,475 interactions.
- Verified zero student overlap between training and test sets.

## 4. Five-Fold Student-Grouped Cross-Validation

**Status: Completed**

- Divided the 80% training pool into five folds.
- Used `KFold` with shuffling and random seed 42.
- Each student belongs to exactly one validation fold.
- Every fold uses four groups for training and one group for validation.
- Verified zero student overlap within every round.
- Created and stored the corresponding interaction DataFrames.

## 5. Reproducibility and Saved Files

**Status: Completed**

- Saved student-level split assignments to:

`data/processed/split_assignments.csv`

- The same student assignments and folds will be reused across all model versions.

## 6. Model Development

**Status: Not Started**

Planned progressive experiments:

| Version | Planned features | Status |
|---|---|---|
| V0 | Global baseline | Pending |
| V1 | Overall student history | Pending |
| V2 | Question difficulty | Pending |
| V3 | Same-skill history | Pending |
| V4 | Other-skill / multi-skill history | Pending |

## 7. Upcoming Tasks

1. Implement and evaluate the V0 baseline using five-fold cross-validation.
2. Introduce overall student history features (V1).
3. Introduce question difficulty features (V2).
4. Introduce same-skill history features (V3).
5. Introduce other-skill / multi-skill history features (V4).
6. Compare all model versions using consistent evaluation metrics.
7. Train the selected final model using the complete 80% training pool.
8. Evaluate once on the untouched 20% test set.
9. Prepare the final report and interactive demonstration.

## 8. Current Project Stage

**Completed:** Dataset audit, cleaning, preparation, and student-level data splitting.

**Next milestone:** V0 — Baseline Model.
