# Next Move AI — System Architecture & Technical Specification

This document details the complete end-to-end architecture of **Next Move AI**, spanning raw educational log ingestion, chronological Bayesian feature engineering, 5-fold student-grouped cross-validation, gradient-boosted decision tree training, threshold optimization, and real-time client-side Float32 browser inference.

---

## 1. Architectural Overview

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        1. RAW INGESTION LAYER                          │
│   ASSISTments 2009-2010 Skill Builder Benchmark (346,860 interactions) │
│      └─ Filter original=1 main questions (275,458 records retained)     │
│      └─ Multi-skill string parsing to Python sets                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  2. STUDENT-LEVEL PARTITIONING LAYER                   │
│   4,217 Unique Students Partitioned (Zero Student Overlap)             │
│      ├─ 80% Training Pool: 3,373 students (219,983 interactions)       │
│      │     └─ 5-Fold Student-Grouped CV (KFold, seed=42)               │
│      └─ 20% Held-Out Test Set: 844 students (55,475 interactions)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│             3. CHRONOLOGICAL BAYESIAN FEATURE ENGINEERING              │
│   Strict sequence ordering: order_id(prior) < order_id(target)          │
│      ├─ G1: Overall Student History (m-estimate, m=10)                 │
│      ├─ G2: Same-Skill History (m=5, multi-skill weight w=1/|K|)       │
│      ├─ G3: Other-Skill Transfer History (m=10)                        │
│      └─ G4: Question Difficulty (m=30, out-of-student prior) + Format  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    4. MODEL TRAINING & SELECTION                       │
│   Scikit-Learn Pipeline: ColumnTransformer + 500-Tree XGBClassifier   │
│      ├─ Cross-Validation: OOF Log Loss=0.5475, Brier=0.1840, AUC=0.7477 │
│      └─ Champion Selection: Depth-wise gradient-boosted decision trees │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│               5. THRESHOLD OPTIMIZATION & RISK ENGINE                  │
│   Calibrate failure risk threshold tau on pooled OOF predictions       │
│      ├─ Risk Probability: r_{s,t} = 1 - P(correct)                     │
│      └─ Operating Threshold tau = 0.2880 -> 75.00% OOF Risk Recall     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│               6. CLIENT-SIDE FLOAT32 BROWSER ENGINE                    │
│   Export 500 trees to JSON (xgboost_trees.json, 2.27 MB)               │
│      ├─ Browser Runtime: Recursive tree traversal with Math.fround     │
│      ├─ Client Feature Synthesizer: smoothing_calculator.js            │
│      └─ Parity: 100% classification concordance to Python XGBoost      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Processing & Leakage Prevention Layer

### 2.1 Problem Framing & Interaction Unit
The prediction unit is an individual student attempting a main question (`original = 1`). Scaffolding sub-questions (`original = 0`, 71,402 records) are excluded from the prediction target pool to ensure consistent evaluation across standard problems.

### 2.2 Student-Grouped Partitioning
To rigorously evaluate model generalization to **unseen students**, partitioning is executed strictly at the student level (`user_id`):
- **Training Pool (80%):** 3,373 students (219,983 interaction records).
- **Held-Out Test Set (20%):** 844 students (55,475 interaction records).
- **Five-Fold Cross-Validation:** The training pool is further partitioned into 5 student-grouped folds using `KFold(n_splits=5, shuffle=True, random_state=42)`. Interactions for any given student appear in exactly one validation fold per round.

---

## 3. Chronological Bayesian Feature Engineering Layer

All feature extraction respects strict chronological temporal ordering: only interactions with global sequence ID $\text{order\_id}_{s,i} < \text{order\_id}_{s,t}$ are visible when constructing features for interaction $t$.

### 3.1 Multi-Skill Credit Attribution
Questions in ASSISTments may map to multiple skills $\mathcal{K}_q = \{k_1, k_2, \dots\}$. To prevent double-counting student success:

$$w_{s,q} = \frac{1}{|\mathcal{K}_q|}$$

Each interaction contributes fractional evidence $w_{s,q}$ to each associated skill, ensuring $\sum_{k \in \mathcal{K}_q} w_{s,q} = 1$.

### 3.2 Empirical Bayesian Shrinkage (m-Estimate Formulation)
When interaction counts are small (cold-start), sample proportions are noisy. We regularize raw empirical rates toward population base rates using empirical Bayesian m-estimates:

$$\hat{\theta} = \frac{C + m \cdot \mu}{N + m}$$

where:
- $C = \sum y_{s,i}$ is the observed correct evidence count (or weighted credit sum).
- $N$ is the observed interaction count (or total exposure weight).
- $m$ is the empirical shrinkage strength parameter (equivalent to pseudo-observations).
- $\mu$ is the baseline population prior estimated strictly on the training fold.

#### Information Groups and Hyperparameters
1. **G1 — Overall Student History:**
   - Feature: `overall_smoothed_accuracy`
   - Parameters: $m = 10$, $\mu \approx 0.6586$ (overall training pool accuracy).
2. **G2 — Same-Skill History:**
   - Features: `weighted_same_skill_smoothed_accuracy`, `log_weighted_same_skill_previous_interactions`
   - Parameters: $m = 5$, $\mu_k$ (skill-specific empirical base rate, falling back to global $\mu$ if unseen).
   - Log transform: $\ln(1 + N_{\text{skill}})$.
3. **G3 — Other-Skill History (Transfer Learning):**
   - Feature: `weighted_other_skill_smoothed_accuracy`
   - Evaluates performance across all skills *other* than the target question's skill(s).
   - Parameters: $m = 10$, $\mu \approx 0.6586$.
4. **G4 — Question Difficulty & Format:**
   - Features: `question_smoothed_accuracy`, `log_question_previous_interactions`, `answer_type`
   - Parameters: $m = 30$, $\mu_k$ (skill base rate). Computed out-of-student to prevent target leakage.
   - Categorical format one-hot encoded: `algebra`, `fill_in_1`, `choose_1`, `choose_n`, `open_response`.

---

## 4. Machine Learning & Training Pipeline

### 4.1 Pipeline Architecture
The pipeline is encapsulated in a scikit-learn `Pipeline`:
```python
ColumnTransformer(
    transformers=[
        ('num', 'passthrough', [
            'overall_smoothed_accuracy',
            'weighted_same_skill_smoothed_accuracy',
            'log_weighted_same_skill_previous_interactions',
            'weighted_other_skill_smoothed_accuracy',
            'question_smoothed_accuracy',
            'log_question_previous_interactions'
        ]),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), [
            'answer_type'
        ])
    ]
)
```
Followed by the champion classifier: `xgboost.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, random_state=42)`.

### 4.2 Cross-Validation Performance
Cross-validation on 219,983 training interactions demonstrated superior performance over baseline and alternative tree models:

| Model Architecture | 5-Fold OOF ROC-AUC | 5-Fold OOF Brier Score | 5-Fold OOF Log Loss | Risk AP (PR-AUC) |
|---|:---:|:---:|:---:|:---:|
| Global Prior Baseline (V0) | 0.5000 | 0.2248 | 0.6418 | 0.3414 |
| Logistic Regression (Combined) | 0.7396 | 0.1868 | 0.5549 | 0.6132 |
| Random Forest (100 Trees) | 0.7468 | 0.1845 | 0.5489 | 0.6261 |
| LightGBM (Leaf-wise GBDT) | 0.7473 | 0.1842 | 0.5479 | 0.6269 |
| **XGBoost (Champion GBDT)** | **0.7477** | **0.1840** | **0.5475** | **0.6277** |

---

## 5. Risk Calibration & Decision Engine

### 5.1 Asymmetric Cost Framing
In educational tutoring, a **false negative** (failing to detect a struggling student) leads to frustration and disengagement. A **false positive** (offering an unnecessary hint) carries minimal operational cost. Therefore, the decision threshold $\tau$ is chosen to prioritize **Risk Recall**:

$$r_{s,t} = 1 - \hat{p}_{s,t} \ge \tau \implies \text{Flag for Pedagogical Assistance}$$

### 5.2 Operating Threshold Selection
Sweeping $\tau \in [0.10, 0.90]$ across the pooled 5-fold OOF predictions selected **$\tau = 0.2880$**:
- Target: 75% Risk Recall.
- Actual 5-Fold OOF Recall: **75.00%** ($56,322 / 75,097$).
- Specificity: **59.82%** ($86,666 / 144,886$).
- Precision: **49.17%** ($56,322 / 114,542$).

---

## 6. Client-Side Browser Inference Architecture

To achieve zero-latency, private, and serverless web interactivity, the trained XGBoost ensemble was adapted for pure client-side execution:

### 6.1 Tree Serialization Format
The 500 decision trees were exported to a structured Float32 JSON schema (`website/data/xgboost_trees.json`, 2.27 MB):
```json
{
  "trees": [
    {
      "nodeid": 0,
      "split": "f1",
      "split_condition": 0.5842,
      "yes": 1,
      "no": 2,
      "children": [...]
    }
  ],
  "base_score": 0.5,
  "learning_rate": 0.03,
  "numeric_features": [...],
  "answer_types": [...]
}
```

### 6.2 Float32 JavaScript Traversal Engine (`website/js/model_inference.js`)
To match XGBoost's C++ floating-point behavior:
1. All split conditions and leaf values are pre-indexed into `Float32Array` buffers using `Math.fround()`.
2. Decision trees are traversed iteratively without recursion overhead.
3. Raw leaf margins are accumulated: $\text{margin} = \sum_{t=1}^{500} \text{leaf}_t(x)$.
4. Final probability is computed via the logistic sigmoid function: $\hat{p} = \frac{1}{1 + e^{-\text{margin}}}$.

### 6.3 Inference Parity Verification
Automated regression tests (`tests/parity/audit_comprehensive_parity.py` and `tests/parity/audit_js_parity.js`) evaluate 16 synthetic boundary test cases across both Python and JavaScript:
- **Max Absolute Error:** $1.905 \times 10^{-7}$
- **Mean Absolute Error:** $7.209 \times 10^{-8}$
- **Binary Classification Concordance ($\tau = 0.2880$):** **100.0% (16/16)**
