import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(".").resolve()))

print("=== COMPREHENSIVE PARITY TEST SUITE GENERATOR ===")

# Load scikit-learn pipeline
pipeline_path = Path("models/final/next_move_xgboost_pipeline.joblib")
if not pipeline_path.exists():
    pipeline_path = Path("notebooks/models/05_combined_model/models/final/next_move_xgboost_pipeline.joblib")
pipe = joblib.load(pipeline_path)

# Define test test cases covering all 7 categories
test_cases = []

# 1. Normal typical cases
normal_samples = [
    # [overall_acc, same_skill_acc, log_same_skill_n, other_skill_acc, q_acc, log_q_n, answer_type]
    [0.725, 0.650, np.log1p(8), 0.740, 0.6626, 2.7081, "algebra"],
    [0.850, 0.900, np.log1p(15), 0.820, 0.5842, 2.5649, "fill_in_1"],
    [0.450, 0.380, np.log1p(5), 0.470, 0.1689, 3.4012, "choose_1"],
    [0.600, 0.620, np.log1p(12), 0.590, 0.8584, 2.8904, "choose_n"],
    [0.674, 0.629, np.log1p(4), 0.674, 0.7586, 2.4849, "open_response"],
]
for i, s in enumerate(normal_samples):
    test_cases.append({"category": "normal", "id": f"normal_{i+1}", "vector": s})

# 2. Student Cold Start (0 total questions, 0 same-skill questions)
# overall_acc = 0.658624, same_skill_acc = 0.658624, log_same = 0.0, other_acc = 0.658624
test_cases.append({
    "category": "cold_start_student",
    "id": "cold_start_1",
    "vector": [0.658624, 0.658624, 0.0, 0.658624, 0.6626, 2.7081, "algebra"]
})
test_cases.append({
    "category": "cold_start_student",
    "id": "cold_start_2",
    "vector": [0.500000, 0.500000, 0.0, 0.500000, 0.500000, 1.0986, "fill_in_1"]
})

# 3. Unseen Skill (Target skill unseen in training pool -> fallback prior = 0.50)
# A student with 10 total questions (7 correct) attempting an unseen skill (0 previous attempts)
# same_skill_acc = (0 + 10 * 0.5) / (0 + 10) = 0.50, log_same = 0.0
# other_skill_acc = (7 + 10 * 0.5) / (10 + 10) = 12 / 20 = 0.60
test_cases.append({
    "category": "unseen_skill",
    "id": "unseen_skill_1",
    "vector": [(7 + 10*0.658624)/20, 0.500000, 0.0, 0.600000, 0.6626, 2.7081, "algebra"]
})
# Student with 2 previous questions on this unseen skill (1 correct)
test_cases.append({
    "category": "unseen_skill",
    "id": "unseen_skill_2",
    "vector": [0.7000, (1 + 10*0.50)/12, np.log1p(2), 0.7200, 0.6000, 2.1972, "choose_1"]
})

# 4. Missing Skill (null skill annotation -> fallback prior = 0.50)
test_cases.append({
    "category": "missing_skill",
    "id": "missing_skill_1",
    "vector": [0.658624, 0.500000, 0.0, 0.500000, 0.6626, 2.7081, "algebra"]
})

# 5. Multi-skill Questions (Skill 16_17, combined prior = 0.525804, other prior = 0.662296)
# Student with 8 previous questions on this multi-skill (4 correct)
test_cases.append({
    "category": "multi_skill",
    "id": "multi_skill_1",
    "vector": [
        (20 + 10*0.658624)/35,
        (4 + 10*0.525804)/(8 + 10),
        np.log1p(8),
        (16 + 10*0.662296)/(27 + 10),
        0.2098,
        3.9318,
        "algebra"
    ]
})

# 6. Unseen Question Cold-Start (0 attempts on this problem across platform)
# question_acc = 0.658624, log_q_exp = 0.0
test_cases.append({
    "category": "unseen_question",
    "id": "unseen_question_1",
    "vector": [0.7500, 0.7000, np.log1p(10), 0.7600, 0.658624, 0.0, "algebra"]
})
test_cases.append({
    "category": "unseen_question",
    "id": "unseen_question_2",
    "vector": [0.4000, 0.3500, np.log1p(4), 0.4200, 0.658624, 0.0, "fill_in_1"]
})

# 7. Boundary values (Extreme accuracies, 0.0 and 1.0, extreme interaction counts)
test_cases.append({
    "category": "boundary",
    "id": "boundary_zero",
    "vector": [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, "algebra"]
})
test_cases.append({
    "category": "boundary",
    "id": "boundary_max",
    "vector": [1.0, 1.0, 5.5, 1.0, 0.99, 5.5, "open_response"]
})
test_cases.append({
    "category": "boundary",
    "id": "boundary_mixed_high_exp",
    "vector": [0.95, 0.10, 5.0, 0.98, 0.50, 4.5, "choose_n"]
})

cols = [
    "overall_smoothed_accuracy",
    "weighted_same_skill_smoothed_accuracy",
    "log_weighted_same_skill_previous_interactions",
    "weighted_other_skill_smoothed_accuracy",
    "question_smoothed_accuracy",
    "log_question_previous_interactions",
    "answer_type"
]

# Run through Python pipeline
py_results = []
for tc in test_cases:
    df_sample = pd.DataFrame([tc["vector"]], columns=cols)
    p_corr = float(pipe.predict_proba(df_sample)[0, 1])
    p_risk = 1.0 - p_corr
    flagged = bool(p_risk >= 0.2880)
    py_results.append({
        "category": tc["category"],
        "id": tc["id"],
        "sample": {c: tc["vector"][j] for j, c in enumerate(cols)},
        "python_prediction": {
            "p_correct": p_corr,
            "p_risk": p_risk,
            "flagged": flagged
        }
    })

out_path = Path(__file__).resolve().parent / "parity_ground_truth.json"
with open(out_path, "w") as f:
    json.dump(py_results, f, indent=2)

print(f"Generated {len(py_results)} test cases covering all 7 audit categories.")
