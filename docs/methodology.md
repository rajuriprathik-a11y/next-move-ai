# Next Move AI — Project Methodology

## 1. Research Objective

The objective of Next Move AI is to predict whether a student's next eligible main question on the ASSISTments 2009–2010 Skill Builder dataset will be recorded as correct, using the student's previously observed interaction history.

This is formulated as a binary classification problem.

- Target variable: `correct`
- `correct = 1`: Recorded correct outcome.
- `correct = 0`: Recorded non-correct outcome.

The prediction must use only information available before the target interaction.

## 2. Dataset Preparation

### 2.1 Dataset

Source: ASSISTments 2009–2010 Skill Builder dataset.

Original dataset dimensions:
- Total interactions: 346,860
- Original columns: 31

### 2.2 Main Question Selection

- Main questions (`original = 1`) are retained as prediction-eligible records.
- Scaffolding questions (`original = 0`) are excluded from prediction targets and historical interaction counters.
- The cleaned dataset contains 275,458 main-question interactions.

### 2.3 Skill ID Processing

- The original `skill_id` column is preserved.
- Multiple skill IDs are parsed into Python sets and stored in `skill_set`.
- Missing skill IDs are represented by empty sets.
- Skill information will be used in later feature-engineering stages.

### 2.4 Chronological Ordering

Interactions are sorted by:

1. `user_id`
2. `order_id`

The `order_id` column is used as the interaction-order identifier. It is not interpreted as a wall-clock timestamp.

For each target interaction, only earlier interactions belonging to the same student may contribute to historical features.

## 3. Data Splitting and Cross-Validation

### 3.1 Student-Level Train-Test Split

The 4,217 unique students are divided into:

| Partition | Percentage | Students |
|---|---:|---:|
| Training pool | 80% | 3,373 |
| Final test set | 20% | 844 |

The split is performed on unique student IDs rather than individual interaction rows.

This ensures that all interactions belonging to a particular student remain in the same partition.

Random seed: `42`.

### 3.2 Five-Fold Student-Grouped Cross-Validation

The 80% training pool is further divided into five approximately equal groups using `KFold` with shuffling and random seed 42.

For each round:

- Four folds are used for training.
- The remaining fold is used for validation.
- Each student appears in a validation fold exactly once.
- No student appears in both training and validation within the same round.

| Round | Training students | Validation students |
|---|---:|---:|
| 1 | 2,698 | 675 |
| 2 | 2,698 | 675 |
| 3 | 2,698 | 675 |
| 4 | 2,699 | 674 |
| 5 | 2,699 | 674 |

The same fold assignments are used for all model versions to ensure fair comparisons.

### 3.3 Final Model Evaluation

After feature selection and hyperparameter tuning through cross-validation:

1. The selected model pipeline is refitted using the complete 80% training pool.
2. The final model is evaluated on the untouched 20% test set.
3. Test results are reported only after model-development decisions are finalized.

The test set is not used for feature selection, hyperparameter tuning, or preprocessing-statistic estimation.

## 4. Feature Engineering Strategy

Features will be introduced progressively to measure their individual contributions.

| Version | Feature group | Description |
|---|---|---|
| V0 | Baseline | Global training-set correct-rate prediction |
| V1 | Overall student history | Historical performance across eligible main questions |
| V2 | Question difficulty | Training-derived question performance estimates |
| V3 | Same-skill history | Historical performance on previously encountered matching skills |
| V4 | Other-skill / multi-skill history | Historical information from other or overlapping skills |

Each version will be evaluated using the same five cross-validation folds.

### 4.1 Historical Feature Construction

For a target interaction at position \(t\), historical features may use only eligible main-question interactions with smaller `order_id` belonging to the same student.

The target interaction and all future interactions are excluded.

Historical interaction count represents the number of previous eligible question records, not the sum of `attempt_count`.

Each previous question contributes at most one binary outcome to historical correct counts.

### 4.2 Smoothed Historical Accuracy

Historical accuracy may be calculated using:

\[
\hat p_t = \frac{C_t+\alpha p_0}{N_t+\alpha}
\]

Where:

- \(C_t\): Number of previous recorded correct outcomes.
- \(N_t\): Number of previous eligible interactions.
- \(p_0\): Correct-rate prior estimated from the relevant training data.
- \(\alpha\): Smoothing parameter.

The prior must be estimated using only the relevant training partition.

### 4.3 Question Difficulty Estimation

Question difficulty estimates are calculated using training-fold outcomes only.

For each cross-validation round, question statistics and the global prior are estimated using that round's four training folds.

Validation outcomes must never contribute to validation difficulty features.

For training rows, target-derived difficulty features must also be constructed without allowing a row