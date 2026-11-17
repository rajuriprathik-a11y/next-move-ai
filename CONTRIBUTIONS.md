# Next Move AI — Team Contributions & Individual Defense Document

**Course:** ES 335: Machine Learning  
**Project:** Next Move AI: Predicting Student Performance Through Learning Interaction History  
**Academic Term:** Monsoon 2026  
**Pair Registration Confirmed:**
- **Rajuri Prathik** (Roll Number: `25110262`)
- **Manikanta Naidu Uppala** (Roll Number: `25110330`)

**Overall Effort Distribution:** **Equal 50% / 50% Partnership**

---

## Executive Statement of Equal Partnership

The Next Move AI system was collaboratively designed, architected, and evaluated by a two-member team. The project workload was partitioned into two technically substantial, complementary, and equally weighted halves:

1. **Rajuri Prathik (50%):** Spearheaded the data engineering, temporal leakage prevention framework, Bayesian $m$-estimate smoothing formulation, student learning history features (Groups G1 & G2), the reusable `src/` Python package, and the client-side Float32 browser inference deployment engine.
2. **Manikanta Naidu Uppala (50%):** Spearheaded the transfer and item difficulty feature engineering (Groups G3 & G4), multimodal feature verification and ablation studies, advanced tree-based ensemble modeling (Random Forest, LightGBM, and XGBoost), asymmetric operational risk decision theory, and holdout test set evaluation.

Both members jointly collaborated on problem formulation, mathematical derivation, cross-validation protocol design, and final report writing.

---

## 1. Individual Profile: Rajuri Prathik (`25110262`)

### 1.1 Specific Contribution & Module Ownership
- **Data Engineering & Temporal Leakage Prevention:**
  - Designed the end-to-end data audit and cleaning pipeline ([01_data_audit.ipynb](notebooks/01_data_audit.ipynb), [02_data_cleaning.ipynb](notebooks/02_data_cleaning.ipynb)), filtering out 71,402 scaffolding records (`original == 0`) to isolate authentic main-question student problem-solving.
  - Implemented the student-disjoint 80/20 train/test partition ([03_data_splitting.ipynb](notebooks/03_data_splitting.ipynb)), ensuring zero student overlap ($\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$) across 4,217 students (3,373 training pool / 844 test).
  - Established strict 1-step lagging (`cumsum().shift(1, fill_value=0)`) and explicitly excluded the post-target `attempt_count` column to prevent forward-looking data leakage.
- **Empirical Bayesian Prior Framework:**
  - Derived the Bayesian $m$-estimate smoothing equation ($\text{Smoothed Accuracy} = \frac{C + \alpha \cdot P}{N + \alpha}$) with smoothing hyperparameter $\alpha = 10$, shrinking noisy small-sample estimates toward chronological population base rates.
- **Student Historical Feature Engineering (Groups G1 & G2):**
  - Developed Group 1 (overall student historical accuracy and logarithmic interaction volume; $V1.1, V1.2$).
  - Developed Group 2 (binary and fractional same-skill historical accuracy; $V2.1–V2.4$), incorporating multi-skill fractional credit attribution ($w_{s,q} = 1/|\mathcal{K}_q|$) when questions map to multiple knowledge components.
- **Core Package & Production Deployment:**
  - Implemented the reusable Python package modules `src/data_loader.py` and `src/feature_engineering.py`.
  - Built the production Float32 JSON decision tree exporter and the client-side JavaScript inference engine powering the interactive Next Move AI web simulator on GitHub Pages.

### 1.2 Key Technical Choices & Justifications
1. **Student-Level Partitioning over Interaction-Level Partitioning:**  
   *Reasoning:* Splitting interaction records uniformly at random allows interactions from the same student to appear in both training and test sets. Because student baseline proficiency is strongly persistent, interaction-level splits measure student re-identification rather than true out-of-sample skill acquisition. Partitioning by unique `user_id` strictly evaluates model generalization to unseen learners.
2. **Bayesian Smoothing ($\alpha = 10$) over Raw Empirical Proportions:**  
   *Reasoning:* A student answering their first question correctly has an empirical rate of $1/1 = 100\%$, while an initial failure yields $0/1 = 0\%$. Using unsmoothed ratios introduces extreme variance for cold-start learners. Shrinking estimates toward the global chronological prior $P$ prevents catastrophic log loss penalties on early interactions.
3. **Client-Side Float32 Tree Inference in JavaScript:**  
   *Reasoning:* Rather than deploying a server-side Python container that introduces latency, hosting cost, and API downtime, model weights were compiled into structured Float32 arrays (`Math.fround()`), enabling instantaneous client-side evaluation directly inside the user's browser.

### 1.3 Evaluation Evidence
- **Empirical Baseline Comparison:** Group 1 features reduced cross-validation Log Loss from $0.6418$ (V0 global prior baseline) to $0.6083$. Incorporating Group 2 fractional same-skill history further drove Log Loss down to $0.5732$, confirming that skill-specific historical memory significantly outperforms aggregate history alone.
- **Parity Test Suite:** Executed automated numerical regression tests comparing Python XGBoost predictions against the JavaScript browser inference engine across 16 boundary scenarios (`tests/parity/`), achieving an exact binary concordance of **100.0%** and a maximum floating-point discrepancy of only $1.905 \times 10^{-7}$.

### 1.4 Limitations
- **Uniform Temporal Weighting:** The feature engineering architecture treats past interactions as uniformly informative, lacking an explicit temporal decay or forgetting curve (e.g., exponential half-life) to account for multi-week learning gaps.
- **Cold-Start Reliance:** For students with zero prior platform interactions ($N=0$), the smoothed features collapse entirely to population priors, limiting personalized risk differentiation on their very first platform attempt.

### 1.5 Reasoning About a Small Change
1. **Scenario A: Reducing the smoothing parameter from $\alpha = 10$ to $\alpha = 1$.**  
   *Expected Impact:* Early-interaction estimates would oscillate wildly between $0.0$ and $1.0$. Cold-start calibration would severely degrade, causing large log loss spikes on early student encounters. While mature students ($N > 50$) would experience negligible changes, the overall cross-validation Brier score would worsen due to overconfidence on small sample sizes.
2. **Scenario B: Replacing student-grouped splitting with standard random interaction splitting.**  
   *Expected Impact:* Cross-validation Log Loss would falsely appear to plummet from $\approx 0.547$ to below $0.480$. However, this performance would be completely spurious—the model would learn student-specific bias terms rather than generalized educational features, failing catastrophically when deployed in production to new students.

---

## 2. Individual Profile: Manikanta Naidu Uppala (`25110330`)

### 2.1 Specific Contribution & Module Ownership
- **Item & Transfer Feature Engineering (Groups G3 & G4):**
  - Designed and implemented Group 3 cross-skill transfer features ($V3.1–V3.4$), quantifying performance on disjoint skills to capture generalized student learning capacity across curricular domains.
  - Developed Group 4 chronological question difficulty features ($V4.1–V4.3$), formulating a leave-one-out historical counter that systematically isolates and subtracts same-student attempts to prevent item-level self-leakage.
  - Engineered question format representations incorporating categorical `answer_type` categories (`algebra`, `fill_in_1`, `choose_1`, `choose_n`, `open_response`).
- **Multimodal Feature Synthesis & Systematic Ablation:**
  - Implemented the automated feature verification pipeline ([00_feature_verification.ipynb](notebooks/models/05_combined_model/00_feature_verification.ipynb)), validating interaction alignment, MultiIndex integrity, and target label consistency across all information groups.
  - Executed the comprehensive feature group ablation study ([01_combined_model_training.ipynb](notebooks/models/05_combined_model/01_combined_model_training.ipynb)), proving the individual and marginal contributions of G1, G2, G3, and G4.
- **Tree-Based Ensemble Modeling & Benchmarking:**
  - Developed, tuned, and cross-validated advanced non-linear architectures: Random Forest, LightGBM, and XGBoost ([02_tree_based_models.ipynb](notebooks/models/05_combined_model/02_tree_based_models.ipynb), [03_final_model_comparison.ipynb](notebooks/models/05_combined_model/03_final_model_comparison.ipynb)).
- **Asymmetric Operational Decision Theory & Final Test Evaluation:**
  - Formulated the risk prediction paradigm ($r_{s,t} = 1 - \hat{p}_{s,t}$) and calibrated the optimal operational decision threshold ($\tau = 0.2880$) targeting $\ge 75\%$ Risk Recall under asymmetric pedagogical intervention costs.
  - Refit the optimal XGBoost pipeline on the full 219,983 training interaction pool and evaluated performance on the completely untouched 55,475 test interactions ([04_final_model_training_and_test_evaluation.ipynb](notebooks/models/05_combined_model/04_final_model_training_and_test_evaluation.ipynb), [comprehensive_evaluation.ipynb](notebooks/models/models_evaluation/comprehensive_evaluation.ipynb)).

### 2.2 Key Technical Choices & Justifications
1. **Selecting XGBoost over Logistic Regression and Random Forest:**  
   *Reasoning:* While linear models effectively capture monotonic skill trends, they fail to model complex interactions between student mastery level and question format (e.g., an algebraic open-response question vs. a multiple-choice item). XGBoost achieved the best out-of-fold Log Loss ($0.5471$) and ROC-AUC ($0.7497$), significantly outperforming Logistic Regression ($0.5636$) and Random Forest ($0.5518$).
2. **Calibrating an Asymmetric Decision Threshold ($\tau = 0.2880$) instead of $0.5000$:**  
   *Reasoning:* In educational tutoring, false negatives (failing to detect a struggling student) result in severe frustration and potential dropout. Conversely, false positives (delivering an unneeded scaffolding hint or review problem) carry near-zero pedagogical penalty. Tuning the threshold to prioritize Risk Recall ($\approx 75\%$) aligns the model directly with educational utility.
3. **Leave-One-Out Question Difficulty Isolation:**  
   *Reasoning:* If a student encounters the same question multiple times, including their own historical attempts in that question's difficulty metric introduces leakage. Dynamically subtracting same-student attempts ensures that item difficulty reflects strictly peer historical performance.

### 2.3 Evaluation Evidence
- **Systematic Ablation Evidence:** Proved through ablation that removing Question Difficulty (G4) causes the single largest degradation in model performance (Log Loss increases from $0.5471$ to $0.5694$), establishing item difficulty as the most critical individual predictive signal.
- **Held-Out Test Set Verification:** On the final held-out test set of 55,475 interactions from 844 unseen students, the final production pipeline verified:
  - **Test Log Loss:** `0.549723` (matching 5-fold CV within $0.0026$, proving zero overfitting)
  - **Test Brier Score:** `0.184909`
  - **Test ROC-AUC:** `0.746778`
  - **Test Risk Recall:** `74.72%` at operating threshold $\tau = 0.2880$ (14,203 out of 19,008 struggling interactions detected)

### 2.4 Limitations
- **Scalar Representation of Sequences:** Gradient boosted decision trees do not model sequential recurrent state transitions directly (unlike Deep Knowledge Tracing / LSTMs); temporal dependencies are compressed into cumulative statistical summary features.
- **Threshold Portability:** The operational decision threshold $\tau = 0.2880$ is specifically calibrated to the ASSISTments baseline failure rate ($\approx 34\%$). Deploying the system to a curriculum with substantially higher or lower failure rates would require threshold recalibration.

### 2.5 Reasoning About a Small Change
1. **Scenario A: Operating at standard default threshold $\tau = 0.5000$ instead of $\tau = 0.2880$.**  
   *Expected Impact:* Risk Recall would drop drastically from $74.72\%$ to approximately $35\%$. The system would fail to detect nearly two out of every three struggling interactions, severely undermining its practical utility as an early intervention support tool.
2. **Scenario B: Omitting Question Difficulty (Group G4) from the feature set.**  
   *Expected Impact:* The model would become completely "item-blind," treating an advanced multi-step algebra problem identically to a trivial arithmetic question. As demonstrated in ablation, Log Loss would worsen by $+0.0223$, and the system would generate widespread false alarms on difficult questions attempted by capable students.

---

## 3. Comprehensive Responsibility & Module Matrix

| Project Component / Notebook / File | Primary Owner | Secondary Contributor | Verification & Defense Role |
|---|:---:|:---:|---|
| `notebooks/01_data_audit.ipynb` | **R. Prathik** | M. N. Uppala | Data distribution, missing value auditing & schema checks |
| `notebooks/02_data_cleaning.ipynb` | **R. Prathik** | M. N. Uppala | Filtering `original == 1`, multi-skill parsing, leakage audit |
| `notebooks/03_data_splitting.ipynb` | **R. Prathik** | M. N. Uppala | 80/20 student-level split, 5-fold student-grouped cross-validation |
| `notebooks/models/00_baseline/baseline.ipynb` | **R. Prathik** | M. N. Uppala | V0 majority class & global prior benchmark metrics |
| `notebooks/models/01_overall_student_history/` (V1.1, V1.2) | **R. Prathik** | M. N. Uppala | Overall smoothed accuracy & logarithmic experience features |
| `notebooks/models/02_same_skill_student_history/` (V2.1–V2.4) | **R. Prathik** | M. N. Uppala | Same-skill binary & fractional attribution models |
| `notebooks/models/03_other_skill_student_history/` (V3.1–V3.4) | **M. N. Uppala** | R. Prathik | Cross-skill transfer representations & models |
| `notebooks/models/04_question_difficulty/` (V4.1–V4.3) | **M. N. Uppala** | R. Prathik | Question difficulty, exposure & categorical answer format |
| `notebooks/models/05_combined_model/00_feature_verification.ipynb` | **M. N. Uppala** | R. Prathik | MultiIndex alignment, assertion checks & target reconciliation |
| `notebooks/models/05_combined_model/01_combined_model_training.ipynb` | **M. N. Uppala** | R. Prathik | Baseline combined logistic regression & feature group ablation |
| `notebooks/models/05_combined_model/02_tree_based_models.ipynb` | **M. N. Uppala** | R. Prathik | Random Forest, LightGBM, and XGBoost 5-fold training |
| `notebooks/models/05_combined_model/03_final_model_comparison.ipynb` | **M. N. Uppala** | R. Prathik | Out-of-fold comparative evaluation, ROC & calibration curves |
| `notebooks/models/05_combined_model/04_final_model_training_and_test_evaluation.ipynb` | **M. N. Uppala** | R. Prathik | Refit pipeline on full training pool & 20% test set evaluation |
| `notebooks/models/models_evaluation/comprehensive_evaluation.ipynb` | **M. N. Uppala** | R. Prathik | Cross-model summary tables across all information groups |
| `src/data_loader.py` & `src/feature_engineering.py` | **R. Prathik** | M. N. Uppala | Modular Python package & vectorized lookup optimizations |
| `website/` (HTML, CSS, Float32 JSON & JS Simulator Engine) | **R. Prathik** | M. N. Uppala | Interactive browser inference simulator on GitHub Pages |
| `tests/parity/` & `tests/test_environment_and_inference.py` | **R. Prathik** | M. N. Uppala | Parity test suite & reproducible environment validation |
| `docs/architecture.md` & `README.md` | **Both** | **Both** | Pipeline architecture documentation, math rendering & guides |

---

## 4. Viva / Defense Preparation Cheatsheet

### Question 1: "Why did you use Bayesian smoothing ($\alpha = 10$) instead of laplace smoothing ($\alpha = 1$)?"
- **Defense by Rajuri Prathik:** Laplace smoothing ($+1$) assumes a uniform prior ($0.5$), which introduces strong negative bias in educational datasets where the average success rate is $\approx 65.8\%$. Our empirical Bayesian smoothing uses $\alpha = 10$ and shrinks toward the *chronological global prior* $P$. At $N=0$, the estimate cleanly equals $P$; with $\alpha = 10$, a student needs approximately 10 interactions for their personal performance to carry equal weight with the population prior, preventing premature overconfidence.

### Question 2: "Why is XGBoost better suited for this task than Logistic Regression or Deep Neural Networks?"
- **Defense by Manikanta Naidu Uppala:** Logistic Regression is constrained by linear additivity; it cannot capture the interaction where an algebraic format penalizes low-mastery students much more severely than high-mastery students without manual interaction terms. XGBoost automatically discovers these non-linear decision boundaries via histogram-based gradient boosted trees (`tree_method="hist"`), yielding the lowest Log Loss ($0.5471$) and superior calibration without requiring millions of parameters like recurrent neural networks.

### Question 3: "How do you guarantee that there is no temporal leakage in your question difficulty feature?"
- **Defense by Both:** Question difficulty for interaction $t$ of student $s$ on problem $q$ considers only interactions with strictly smaller sequence IDs ($\text{order\_id} < \text{order\_id}_{s,t}$). Crucially, we maintain a secondary student-problem index and subtract all historical attempts made by student $s$ themselves on problem $q$. This guarantees that student $s$'s past success or struggle on that exact question cannot leak into their question difficulty estimate.
