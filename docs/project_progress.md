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

## 6. Model Development Lifecycle & Experimental Milestones

**Status: Completed**

The project completed a systematic, staged feature engineering and modeling progression across five distinct information groups, followed by multi-model exploration, threshold optimization, and final test evaluation.

### Summary of Completed Milestones

| Milestone / Experiment | Features Engineered & Evaluated | 5-Fold OOF ROC-AUC | 5-Fold OOF Brier Score | 5-Fold OOF Log Loss | Status |
|---|---|:---:|:---:|:---:|:---:|
| **V0: Global Prior Baseline** | Unconditional training-fold base rate ($\hat{p} \approx 0.6586$) | 0.5000 | 0.2248 | 0.6418 | Completed |
| **V1: Overall Student History** | Cumulative overall smoothed accuracy ($m=10$) + interaction count | 0.6739 | 0.2035 | 0.5944 | Completed |
| **V2: Same-Skill History** | Cumulative target-skill smoothed accuracy ($m=5$) + fractional multi-skill weighting | 0.6862 | 0.2021 | 0.5903 | Completed |
| **V3: Other-Skill History** | Cumulative non-target-skill transfer accuracy ($m=10$) | 0.6267 | 0.2137 | 0.6170 | Completed |
| **V4: Question Difficulty** | Item empirical difficulty ($m=30$) + log exposure + categorical format | 0.6356 | 0.2127 | 0.6150 | Completed |
| **Combined Feature Set** | 7 unified features across G1–G4 (11 transformed dimensions) | 0.7396 | 0.1868 | 0.5549 | Completed |
| **Ablation Study (Logistic Regression)** | Full model vs dropping each individual information group | 0.7396 | 0.1868 | 0.5549 | Completed |
| **Tree-Based Models (Random Forest)** | Non-linear ensemble with bagging and bootstrapping | 0.7468 | 0.1845 | 0.5489 | Completed |
| **Tree-Based Models (LightGBM)** | Gradient boosting with histogram binning and leaf-wise growth | 0.7473 | 0.1842 | 0.5479 | Completed |
| **Tree-Based Models (XGBoost Champion)** | Depth-wise tree boosting (500 estimators, depth 5, $\eta=0.03$) | **0.7477** | **0.1840** | **0.5475** | **Champion Selected** |

---

## 7. Operational Threshold Optimization ($\tau = 0.2880$)

**Status: Completed**

To operationalize probability predictions into proactive educational interventions, an operational failure-risk threshold $\tau$ was calibrated on the pooled 5-fold Out-of-Fold (OOF) cross-validation predictions:
- **Optimization Criterion:** Achieve approximately 75% Risk Recall to ensure three out of four struggling student interactions are proactively flagged for instructional assistance.
- **Calibrated Operating Threshold:** $\tau = 0.2880$ (flag for intervention if predicted failure risk $P(\text{incorrect}) \ge 0.2880$, or equivalently $P(\text{correct}) \le 0.7120$).
- **5-Fold OOF Cross-Validation Results at $\tau = 0.2880$:**
  - **Risk Recall:** $75.00\%$ ($56,322$ True Positives / $75,097$ actual risk interactions)
  - **Specificity:** $59.82\%$ ($86,666$ True Negatives / $144,886$ non-risk interactions)
  - **Precision:** $49.17\%$ ($56,322$ True Positives / $114,542$ flagged interactions)
  - **False Positive Rate:** $40.18\%$ ($58,220$ False Positives / $144,886$ non-risk interactions)

---

## 8. Final Champion Retraining & Untouched Held-Out Test Evaluation

**Status: Completed**

The champion XGBoost architecture was refit on the entire 80% training pool ($219,983$ interactions, $3,373$ students) and evaluated exactly once on the untouched 20% held-out test set ($55,475$ interactions, $844$ students):

- **Probabilistic Calibration & Discrimination:**
  - **ROC-AUC:** `0.746778` (demonstrating consistent generalization on unseen students vs 0.747729 OOF CV)
  - **Brier Score:** `0.184909` (minimal probability error increase vs 0.184013 OOF CV)
  - **Log Loss:** `0.549723` (stable probabilistic loss vs 0.547472 OOF CV)
  - **Average Precision (Risk PR-AUC):** `0.624092`
- **Downstream Binary Risk Classification ($\tau = 0.2880$):**
  - **Risk Recall:** `74.72%` ($14,203$ True Positives / $19,008$ actual test risk interactions)
  - **Specificity:** `60.21%` ($21,956$ True Negatives / $36,467$ non-risk test interactions)
  - **Precision:** `49.46%` ($14,203$ True Positives / $28,714$ flagged test interactions)
  - **F1 Score:** `0.5952`
- **Artifacts Saved:**
  - Fitted pipeline: `models/final/next_move_xgboost_pipeline.joblib`
  - Test metrics summary: `models/final/final_evaluation_metrics.json`
  - Interaction test predictions: `results/test_predictions/xgboost_test_predictions.csv`

---

## 9. Interactive Demonstration & Web Deployment

**Status: Completed**

- **Client-Side Float32 Model Serialization:** Extracted 500 decision trees into structured JSON format (`website/data/xgboost_trees.json`, 2.27 MB) for zero-latency in-browser evaluation.
- **Interactive Student Outcome Simulator:** Real-time Bayesian smoothing and recursive tree traversal engine implemented in JavaScript with 100% classification concordance to Python scikit-learn/XGBoost.
- **Production Web Deployment:** Deployed via GitHub Actions to GitHub Pages at:
  `https://rajuriprathik-a11y.github.io/next-move-ai/`

---

## 10. Current Project Stage

**Completed:** All developmental, experimental, validation, and deployment milestones successfully achieved. Repository restructured into a clean, reproducible, production-grade format.
