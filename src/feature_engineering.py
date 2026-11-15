
import numpy as np
import pandas as pd


def add_overall_history_features(data, alpha=10, prior_data=None):
    """
    Construct chronological overall student-history features.

    Features:
    1. Smoothed historical accuracy
    2. Log-transformed previous interaction count

    The global smoothing prior uses only eligible interactions
    with a smaller order_id.
    """

    df = data.copy()

    # Ensure chronological order within each student
    df = df.sort_values(
        ["user_id", "order_id"]
    ).reset_index(drop=True)

    # 1. Count previous interactions of the same student
    df["previous_interactions"] = (
        df.groupby("user_id").cumcount()
    )

    # 2. Count previous correct interactions of the same student
    df["previous_correct"] = (
        df.groupby("user_id")["correct"]
          .transform(
              lambda s: s.cumsum().shift(1, fill_value=0)
          )
    )

    # 3. Select the data allowed to contribute to the global prior
    if prior_data is None:
        prior_data = data

    prior_df = prior_data.sort_values(
        "order_id"
    )

    prior_order_ids = prior_df["order_id"].to_numpy()
    prior_correct = prior_df["correct"].to_numpy()

    # 4. Cumulative global correct answers
    cumulative_correct = np.concatenate(
        ([0], np.cumsum(prior_correct))
    )

    # Find the number of prior rows with smaller order_id
    positions = np.searchsorted(
        prior_order_ids,
        df["order_id"].to_numpy(),
        side="left"
    )

    # 5. Calculate the past-only global prior
    global_prior = np.full(len(df), 0.5)

    has_history = positions > 0

    global_prior[has_history] = (
        cumulative_correct[positions[has_history]]
        / positions[has_history]
    )

    df["global_historical_prior"] = global_prior

    # 6. V1.1: Smoothed historical accuracy
    df["overall_smoothed_accuracy"] = (
        df["previous_correct"] + alpha * global_prior
    ) / (
        df["previous_interactions"] + alpha
    )

    # 7. V1.2: Log-transformed previous interaction count
    df["log_previous_interactions"] = np.log1p(
        df["previous_interactions"]
    )

    return df


from bisect import bisect_left


def add_same_skill_history_features(data, alpha=10, prior_data=None):
    """
    Generate binary and fractional same-skill history features.

    Binary:
        A previous question contributes 1 if at least one skill overlaps.

    Fractional:
        A previous question contributes:
        number of overlapping skills / number of skills in that question.

    Student history:
        Only earlier questions from the same student in `data`.

    Global prior:
        All eligible earlier questions in `prior_data` whose skill sets
        overlap with the current target question.
    """

    df = data.copy()

    df = df.sort_values(
        ["user_id", "order_id"]
    ).reset_index(drop=True)

    # Convert skill information into Python sets.
    def get_skill_set(row):
        if "skill_set" in row.index:
            value = row["skill_set"]

            if isinstance(value, set):
                return value

        value = row["skill_id"]

        if pd.isna(value):
            return set()

        return set(str(value).split("_"))

    skill_sets = [
        get_skill_set(row)
        for _, row in df.iterrows()
    ]

    outcomes = df["correct"].to_numpy()
    order_ids = df["order_id"].to_numpy()

    n = len(df)

    # --------------------------------------------------
    # 1. Build the chronological global-prior index
    # --------------------------------------------------

    if prior_data is None:
        prior_data = data

    prior_df = prior_data.copy()
    prior_df = prior_df.sort_values("order_id").reset_index(drop=True)

    prior_skill_sets = [
        get_skill_set(row)
        for _, row in prior_df.iterrows()
    ]

    prior_order_ids = prior_df["order_id"].to_numpy()
    prior_outcomes = prior_df["correct"].to_numpy()

    # For each skill, store the positions of earlier questions
    # containing that skill.
    skill_prior_positions = {}

    for pos, skills in enumerate(prior_skill_sets):
        for skill in skills:
            skill_prior_positions.setdefault(
                skill, []
            ).append(pos)

    # Store the chronological global prior for every target row.
    global_prior = np.full(n, 0.5)

    for pos, current_skills in enumerate(skill_sets):

        matched_prior_positions = set()

        for skill in current_skills:

            skill_positions = skill_prior_positions.get(
                skill, []
            )

            # Find positions having order_id strictly smaller
            # than the current target's order_id.
            cutoff = bisect_left(
                prior_order_ids[skill_positions],
                order_ids[pos]
            ) if skill_positions else 0

            matched_prior_positions.update(
                skill_positions[:cutoff]
            )

        if matched_prior_positions:
            matched_positions = list(matched_prior_positions)

            global_prior[pos] = prior_outcomes[
                matched_positions
            ].mean()

    # --------------------------------------------------
    # 2. Calculate the student's previous same-skill history
    # --------------------------------------------------

    binary_experience = np.zeros(n)
    binary_correct = np.zeros(n)

    fractional_experience = np.zeros(n)
    fractional_correct = np.zeros(n)

    for positions in df.groupby(
        "user_id", sort=False
    ).indices.values():

        skill_question_ids = {}

        fractional_skill_experience = {}
        fractional_skill_correct = {}

        for pos in positions:

            current_skills = skill_sets[pos]

            # Find unique previous questions sharing at least
            # one skill with the current question.
            matched_question_ids = set()

            for skill in current_skills:
                matched_question_ids.update(
                    skill_question_ids.get(skill, set())
                )

            # Binary history: each previous question counted once.
            binary_experience[pos] = len(
                matched_question_ids
            )

            binary_correct[pos] = sum(
                outcomes[qid]
                for qid in matched_question_ids
            )

            # Fractional history: sum contributions from
            # the skills shared with the current question.
            fractional_experience[pos] = sum(
                fractional_skill_experience.get(skill, 0.0)
                for skill in current_skills
            )

            fractional_correct[pos] = sum(
                fractional_skill_correct.get(skill, 0.0)
                for skill in current_skills
            )

            # Add current question to history only AFTER
            # calculating its features.
            if current_skills:

                weight = 1.0 / len(current_skills)

                for skill in current_skills:

                    skill_question_ids.setdefault(
                        skill, set()
                    ).add(pos)

                    fractional_skill_experience[skill] = (
                        fractional_skill_experience.get(skill, 0.0)
                        + weight
                    )

                    fractional_skill_correct[skill] = (
                        fractional_skill_correct.get(skill, 0.0)
                        + weight * outcomes[pos]
                    )

    # --------------------------------------------------
    # 3. Store historical count features
    # --------------------------------------------------

    df["same_skill_previous_interactions"] = binary_experience
    df["same_skill_previous_correct"] = binary_correct

    df["weighted_same_skill_previous_interactions"] = fractional_experience
    df["weighted_same_skill_previous_correct"] = fractional_correct

    # --------------------------------------------------
    # 4. Store chronological global prior
    # --------------------------------------------------

    df["same_skill_global_prior"] = global_prior

    # --------------------------------------------------
    # 5. Calculate smoothed accuracy
    # --------------------------------------------------

    df["same_skill_smoothed_accuracy"] = (
        binary_correct + alpha * global_prior
    ) / (
        binary_experience + alpha
    )

    df["weighted_same_skill_smoothed_accuracy"] = (
        fractional_correct + alpha * global_prior
    ) / (
        fractional_experience + alpha
    )

    # --------------------------------------------------
    # 6. Log-transformed experience features
    # --------------------------------------------------

    df["log_same_skill_previous_interactions"] = np.log1p(
        binary_experience
    )

    df["log_weighted_same_skill_previous_interactions"] = np.log1p(
        fractional_experience
    )

    return df