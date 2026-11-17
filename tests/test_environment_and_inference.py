"""
Next Move AI — Clean Environment & Model Inference Integration Test
Verifies dependency imports, scikit-learn pipeline loading, and end-to-end Float64/Float32 prediction accuracy.
"""

import sys
from pathlib import Path
import json

print("=== NEXT MOVE AI ENVIRONMENT & INFERENCE VERIFICATION ===")

# 1. Dependency imports test
print("\n[1/4] Verifying Core Dependency Imports...")
try:
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    import xgboost
    import lightgbm
    import joblib
    import matplotlib
    try:
        import seaborn
        seaborn_ver = seaborn.__version__
    except ImportError:
        seaborn_ver = "Not installed (optional visualization)"
    print(f"  ✓ numpy:        {np.__version__}")
    print(f"  ✓ pandas:       {pd.__version__}")
    print(f"  ✓ scipy:        {scipy.__version__}")
    print(f"  ✓ scikit-learn: {sklearn.__version__}")
    print(f"  ✓ xgboost:      {xgboost.__version__}")
    print(f"  ✓ lightgbm:     {lightgbm.__version__}")
    print(f"  ✓ joblib:       {joblib.__version__}")
    print(f"  ✓ matplotlib:   {matplotlib.__version__}")
    print(f"  ✓ seaborn:      {seaborn_ver}")
except ImportError as e:
    print(f"  ✗ FAILED to import dependencies: {e}")
    sys.exit(1)

# 2. Source package imports test
print("\n[2/4] Verifying Project Source Package Imports...")
try:
    sys.path.insert(0, str(Path(".").resolve()))
    import src
    from src import data_loader
    from src import feature_engineering
    print("  ✓ src package import successful")
    print("  ✓ src.data_loader import successful")
    print("  ✓ src.feature_engineering import successful")
except ImportError as e:
    print(f"  ✗ FAILED to import src modules: {e}")
    sys.exit(1)

# 3. Serialized model loading test
print("\n[3/4] Loading Serialized Pipeline & Metadata Artifacts...")
model_path = Path("models/final/next_move_xgboost_pipeline.joblib")
metrics_path = Path("models/final/final_evaluation_metrics.json")

assert model_path.exists(), f"Model file missing: {model_path}"
assert metrics_path.exists(), f"Metrics file missing: {metrics_path}"

pipeline = joblib.load(model_path)
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

print(f"  ✓ Successfully loaded pipeline from: {model_path}")
print(f"  ✓ Pipeline steps: {[name for name, _ in pipeline.steps]}")
print(f"  ✓ Final test ROC-AUC: {metrics['test_metrics']['roc_auc']:.6f}")
print(f"  ✓ Final test Brier:   {metrics['test_metrics']['brier_score']:.6f}")
print(f"  ✓ Operating threshold: {metrics['threshold']}")

# 4. Sample inference & threshold decision test
print("\n[4/4] Executing Sample Inference on Curated Presets...")
test_scenarios = [
    {
        "name": "Cold-Start Learner (N=0)",
        "features": {
            "overall_smoothed_accuracy": 0.6586,
            "weighted_same_skill_smoothed_accuracy": 0.6586,
            "log_weighted_same_skill_previous_interactions": 0.0,
            "weighted_other_skill_smoothed_accuracy": 0.6586,
            "question_smoothed_accuracy": 0.5842,
            "log_question_previous_interactions": 2.5649,
            "answer_type": "algebra"
        },
        "expected_risk_above_tau": True
    },
    {
        "name": "High Achiever (N=60, C=54)",
        "features": {
            "overall_smoothed_accuracy": 0.8655,
            "weighted_same_skill_smoothed_accuracy": 0.9097,
            "log_weighted_same_skill_previous_interactions": 2.7726,
            "weighted_other_skill_smoothed_accuracy": 0.8529,
            "question_smoothed_accuracy": 0.7250,
            "log_question_previous_interactions": 3.1500,
            "answer_type": "algebra"
        },
        "expected_risk_above_tau": False
    },
    {
        "name": "Strong Student, Weak Skill (Target Skill struggle)",
        "features": {
            "overall_smoothed_accuracy": 0.7561,
            "weighted_same_skill_smoothed_accuracy": 0.4058,
            "log_weighted_same_skill_previous_interactions": 2.1972,
            "weighted_other_skill_smoothed_accuracy": 0.8635,
            "question_smoothed_accuracy": 0.6643,
            "log_question_previous_interactions": 3.2189,
            "answer_type": "fill_in_1"
        },
        "expected_risk_above_tau": True
    }
]

tau = metrics["threshold"]
all_passed = True

for sc in test_scenarios:
    df_sample = pd.DataFrame([sc["features"]])
    p_correct = float(pipeline.predict_proba(df_sample)[0, 1])
    p_risk = 1.0 - p_correct
    flagged = bool(p_risk >= tau)
    match = (flagged == sc["expected_risk_above_tau"])
    status = "PASS" if match else "FAIL"
    if not match:
        all_passed = False
    print(f"  Scenario: {sc['name']}")
    print(f"    P(Correct): {p_correct*100:.2f}% | Risk P(incorrect): {p_risk*100:.2f}% | Flagged: {flagged} [Status: {status}]")

assert all_passed, "One or more scenario predictions failed expected classification!"
print("\n>>> ALL ENVIRONMENT & INFERENCE TESTS PASSED SUCCESSFULLY! <<<\n")
