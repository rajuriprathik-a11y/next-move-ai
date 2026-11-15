# Data Cleaning and Preprocessing

## 1. Dataset Information

**Dataset:** ASSISTments 2009–2010 Skill Builder  
**Source file:** `skill_builder_data_corrected_collapsed.csv`

Original dataset dimensions:
- Rows: 346,860
- Columns: 31

## 2. Main Question Selection

The `original` column distinguishes main questions from scaffolding questions.

| Question type | original | Rows |
|---|---:|---:|
| Main questions | 1 | 275,458 |
| Scaffolding questions | 0 | 71,402 |

**Decision:** Retain main questions as prediction-eligible records for the initial experiment. Scaffolding questions are excluded as prediction targets.

## 3. Missing Value Analysis

| Column | Finding |
|---|---|
| `user_id` | No missing values |
| `order_id` | No missing values |
| `problem_id` | No missing values |
| `correct` | No missing values |
| `skill_id` | 16,059 missing values among main questions (5.83%) |

**Decision:** Retain rows with missing skill information. They remain useful for overall student-history features but cannot contribute to skill-specific history.

## 4. Target Variable Validation

The target variable is `correct`.

| Value | Meaning in dataset | Main-question rows |
|---|---|---:|
| 0 | Recorded non-correct outcome | 94,105 |
| 1 | Recorded correct outcome | 181,353 |

No missing target values were found.

The target remains binary.

## 5. Skill ID Parsing

Created a new column named `skill_set`.

- Split multi-skill identifiers using `_`.
- Converted skill identifiers into Python sets.
- Represented missing skill IDs using empty sets.
- Preserved the original `skill_id` column.

Results:
- Known skill information: 259,399 rows.
- Missing skill information: 16,059 rows.
- Missing-value and empty-set consistency validation passed.

## 6. Chronological Ordering

Validation results:
- `order_id` data type: `int64`.
- Unique order IDs: 275,458.
- Duplicate order IDs: 0.
- Students checked: 4,217.
- Students with increasing order IDs: 4,217.
- Students with ordering violations: 0.

**Decision:** Sort interactions by `user_id` and `order_id` in ascending order.

This establishes the interaction ordering used for temporal feature construction.

## 7. Attempt Count Investigation

The `attempt_count` column represents the number of answer submissions within an individual question.

Important distinction:
- `attempt_count` describes answer submissions for one question.
- Historical interaction count in our feature engineering represents the number of previous eligible question rows.

**Decisions:**
- Retain zero-attempt and multiple-attempt records.
- Preserve recorded `correct` values.
- Never use the target row's `attempt_count` as a predictor.
- Count historical interactions using eligible question rows, not the sum of `attempt_count`.

## 8. Initial Cleaned Dataset

Created a focused DataFrame named `clean_df` containing:

1. `user_id`
2. `order_id`
3. `problem_id`
4. `correct`
5. `skill_id`
6. `skill_set`
7. `attempt_count`

The original raw DataFrame was preserved.

**Processed dataset:**
- Rows: 275,458
- Columns: 7
- Sorting: By `user_id` and `order_id`
- Output: `data/processed/main_questions_cleaned.csv`

## 9. Current Status

Initial data filtering, skill parsing, chronological validation, and creation of the focused processed dataset are complete.

The next stage is student-disjoint data splitting and baseline model evaluation.