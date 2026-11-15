
import numpy as np
import pandas as pd

def add_overall_history_features(data, p0, alpha=10, training=False):
    """
    Construct chronological overall student-history features.

    Features:
    1. Smoothed historical accuracy
    2. Log-transformed previous interaction count
    """

    df = data.copy()

    # Ensure chronological order
    df = df.sort_values(
        ["user_id", "order_id"]
    ).copy()

    # Count previous interactions
    df["previous_interactions"] = (
        df.groupby("user_id").cumcount()
    )

    # Count previous correct interactions
    df["previous_correct"] = (
        df.groupby("user_id")["correct"]
          .transform(
              lambda s: s.cumsum().shift(1, fill_value=0)
          )
    )

    # Calculate global prior
    if training:
        total_correct = df["correct"].sum()
        total_interactions = len(df)

        row_prior = (
            total_correct - df["correct"]
        ) / (total_interactions - 1)

    else:
        row_prior = p0

    # V1.1: Smoothed historical accuracy
    df["overall_smoothed_accuracy"] = (
        df["previous_correct"] + alpha * row_prior
    ) / (
        df["previous_interactions"] + alpha
    )

    # V1.2: Log-transformed previous interaction count
    df["log_previous_interactions"] = np.log1p(
        df["previous_interactions"]
    )

    return df

def add_same_skill_history_features(data, p0, alpha=10, training=False):
    """
    Generate binary and fractional same-skill history features.

    Binary:
        A previous question contributes 1 if at least one skill overlaps.

    Fractional:
        A previous question contributes:
        number of overlapping skills / number of skills in that question.

    Only previous interactions from the same student are considered.
    """

    df = data.copy()
    df = df.sort_values(
        ["user_id", "order_id"]
    ).reset_index(drop=True)

    n = len(df)

    # Convert skill IDs into Python sets.
    skill_sets = df["skill_id"].apply(
        lambda x: set(str(x).split("_")) if pd.notna(x) else set()
    ).tolist()

    outcomes = df["correct"].to_numpy()

    # Calculate leakage-safe global prior.
    if training:
        total_correct = outcomes.sum()

        if n > 1:
            row_prior = (total_correct - outcomes) / (n - 1)
        else:
            row_prior = np.full(n, p0)
    else:
        row_prior = np.full(n, p0)

    # Arrays to store calculated features.
    binary_experience = np.zeros(n)
    binary_correct = np.zeros(n)

    fractional_experience = np.zeros(n)
    fractional_correct = np.zeros(n)

    # Process each student independently.
    for positions in df.groupby("user_id", sort=False).indices.values():

        # Previous question IDs associated with each skill.
        skill_question_ids = {}

        # Accumulated fractional experience for each skill.
        skill_fractional_experience = {}
        skill_fractional_correct = {}

        for pos in positions:

            current_skills = skill_sets[pos]

            # Calculate features BEFORE adding current question.
            matched_question_ids = set()

            for skill in current_skills:
                matched_question_ids.update(
                    skill_question_ids.get(skill, set())
                )

            # V2.1 / V2.3: Binary history.
            binary_experience[pos] = len(matched_question_ids)

            binary_correct[pos] = sum(
                outcomes[qid] for qid in matched_question_ids
            )

            # V2.2 / V2.4: Fractional history.
            fractional_experience[pos] = sum(
                skill_fractional_experience.get(skill, 0.0)
                for skill in current_skills
            )

            fractional_correct[pos] = sum(
                skill_fractional_correct.get(skill, 0.0)
                for skill in current_skills
            )

            # Add current question to history only AFTER calculation.
            if current_skills:
                weight = 1.0 / len(current_skills)

                for skill in current_skills:

                    skill_question_ids.setdefault(
                        skill, set()
                    ).add(pos)

                    skill_fractional_experience[skill] = (
                        skill_fractional_experience.get(skill, 0.0)
                        + weight
                    )

                    skill_fractional_correct[skill] = (
                        skill_fractional_correct.get(skill, 0.0)
                        + weight * outcomes[pos]
                    )

    # Store raw historical counts and weighted counts.
    df["same_skill_previous_interactions"] = binary_experience
    df["same_skill_previous_correct"] = binary_correct

    df["weighted_same_skill_previous_interactions"] = fractional_experience
    df["weighted_same_skill_previous_correct"] = fractional_correct

    # Smoothed accuracy features.
    df["same_skill_smoothed_accuracy"] = (
        binary_correct + alpha * row_prior
    ) / (binary_experience + alpha)

    df["weighted_same_skill_smoothed_accuracy"] = (
        fractional_correct + alpha * row_prior
    ) / (fractional_experience + alpha)

    # Log-transformed experience features.
    df["log_same_skill_previous_interactions"] = np.log1p(
        binary_experience
    )

    df["log_weighted_same_skill_previous_interactions"] = np.log1p(
        fractional_experience
    )

    return df

