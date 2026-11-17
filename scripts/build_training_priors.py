import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from src.feature_engineering import _normalise_other_skill_set

print("=== BUILDING TRAINING PRIORS & QUESTION STATISTICS ===")

# 1. Load data
split_path = Path("data/processed/split_assignments.csv")
data_path = Path("data/processed/main_questions_cleaned.csv")

splits = pd.read_csv(split_path)
train_users = set(splits[splits["partition"] == "train_pool"]["user_id"])

df = pd.read_csv(data_path)
train_df = df[df["user_id"].isin(train_users)].copy()
train_df = train_df.sort_values(["user_id", "order_id"]).reset_index(drop=True)

n_total = len(train_df)
c_total = int(train_df["correct"].sum())
print(f"Training pool total rows: {n_total}")
print(f"Training pool total correct: {c_total}")

# 1. G1 Final Global Prior
g1_global_prior = float(c_total / n_total)
g1_cold_start_fallback = 0.50
print(f"G1 Global Prior: {g1_global_prior:.6f}")

# Convert skill_set properly
train_df["normalized_skill_set"] = train_df["skill_id"].apply(_normalise_other_skill_set)

# Annotated mask
annotated_mask = train_df["normalized_skill_set"].apply(lambda s: len(s) > 0)
annotated_df = train_df[annotated_mask]
n_annotated = int(len(annotated_df))
c_annotated = int(annotated_df["correct"].sum())
print(f"Annotated training rows: {n_annotated}")
print(f"Annotated training correct: {c_annotated} ({c_annotated / n_annotated:.6f})")

# 2. Build skill fractional index across training pool
skill_weights = {}
skill_weighted_correct = {}
skill_raw_counts = {}
skill_raw_correct = {}

for skill_set, correct in zip(train_df["normalized_skill_set"], train_df["correct"]):
    if not skill_set:
        continue
    w = 1.0 / len(skill_set)
    for skill in skill_set:
        skill_weights[skill] = skill_weights.get(skill, 0.0) + w
        skill_weighted_correct[skill] = skill_weighted_correct.get(skill, 0.0) + w * correct
        skill_raw_counts[skill] = skill_raw_counts.get(skill, 0) + 1
        skill_raw_correct[skill] = skill_raw_correct.get(skill, 0) + int(correct)

print(f"Total unique individual skills in training pool: {len(skill_weights)}")

# Create skill dictionary
skills_dict = {}
for skill, w in sorted(skill_weights.items()):
    c_w = skill_weighted_correct[skill]
    raw_n = skill_raw_counts[skill]
    raw_c = skill_raw_correct[skill]
    prior = float(c_w / w) if w > 0 else 0.5
    
    # Calculate G3 other-skill prior for this single skill
    other_w = n_annotated - w
    other_c = c_annotated - c_w
    other_prior = float(other_c / other_w) if other_w > 0 else 0.5
    
    skills_dict[str(skill)] = {
        "skill_id": str(skill),
        "fractional_weight": round(w, 4),
        "fractional_correct": round(c_w, 4),
        "empirical_prior": round(prior, 6),
        "raw_count": raw_n,
        "raw_correct": raw_c,
        "other_skill_prior": round(other_prior, 6)
    }

# Test sample skills
print("\nSample skills G2 prior:")
for s in ["27", "76", "18", "92", "310", "16", "17"]:
    if s in skills_dict:
        sd = skills_dict[s]
        print(f"Skill {s}: prior={sd['empirical_prior']:.4f}, weight={sd['fractional_weight']:.1f}, other_prior={sd['other_skill_prior']:.4f}")

# 4. Question level statistics (G4)
# Group by problem_id
q_grouped = train_df.groupby("problem_id")
question_stats = {}

for pid, group in q_grouped:
    q_count = int(len(group))
    q_correct = int(group["correct"].sum())
    # Empirical smoothed accuracy using G1 training prior
    q_smoothed = float((q_correct + 10.0 * g1_global_prior) / (q_count + 10.0))
    q_log_int = float(np.log1p(q_count))
    
    # Representative format and skill
    at = group["answer_type"].iloc[0]
    sid = group["skill_id"].iloc[0]
    sid_str = str(sid) if pd.notna(sid) else None
    
    question_stats[str(pid)] = {
        "problem_id": int(pid),
        "attempts": q_count,
        "correct": q_correct,
        "smoothed_accuracy": round(q_smoothed, 6),
        "log_interactions": round(q_log_int, 6),
        "answer_type": str(at),
        "skill_id": sid_str
    }

print(f"\nTotal unique questions in training pool: {len(question_stats)}")

# Save full priors artifact
training_priors_artifact = {
    "description": "Next Move AI — Training-Only Historical Statistics and Bayesian Smoothing Priors",
    "partition": "80% student-level training pool (strictly no test data)",
    "training_interactions": n_total,
    "training_correct": c_total,
    "annotated_interactions": n_annotated,
    "annotated_correct": c_annotated,
    "smoothing_alpha": 10.0,
    "g1_overall_history": {
        "final_global_prior": round(g1_global_prior, 6),
        "cold_start_fallback": 0.50
    },
    "g2_same_skill_history": {
        "total_skills": len(skills_dict),
        "skill_priors_mean": round(float(np.mean([s["empirical_prior"] for s in skills_dict.values()])), 6),
        "unseen_skill_fallback": 0.50,
        "missing_skill_fallback": 0.50,
        "aggregation_rule": "multi-skill questions sum fractional_correct / sum fractional_weight across target skills",
        "skills": skills_dict
    },
    "g3_other_skill_history": {
        "annotated_pool_count": n_annotated,
        "annotated_pool_correct": c_annotated,
        "unannotated_fallback": 0.50,
        "cold_start_fallback": 0.50,
        "formula": "P_other = (annotated_pool_correct - sum(same_skill_correct)) / (annotated_pool_count - sum(same_skill_weight))"
    },
    "g4_question_difficulty": {
        "total_questions": len(question_stats),
        "unseen_question_fallback": {
            "smoothed_accuracy": round(g1_global_prior, 6),
            "log_interactions": 0.0,
            "attempts": 0,
            "correct": 0
        },
        "sample_questions_summary": {
            "min_attempts": int(min(q["attempts"] for q in question_stats.values())),
            "median_attempts": int(np.median([q["attempts"] for q in question_stats.values()])),
            "max_attempts": int(max(q["attempts"] for q in question_stats.values()))
        }
    }
}

# Write summary priors file (lightweight for website: ~35KB)
output_path = Path("website/data/training_priors.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(training_priors_artifact, f, indent=2)

print(f"\nSaved website/data/training_priors.json ({output_path.stat().st_size / 1024:.1f} KB)")
