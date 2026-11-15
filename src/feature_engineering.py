
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

def _build_fractional_skill_index(data):
    """
    Build chronological cumulative weighted-history indexes.

    Each previous question distributes its contribution equally
    across its own skills: 1 / number_of_previous_question_skills.
    """

    skill_entries = {}

    for order_id, skill_set, correct in zip(
        data["order_id"],
        data["skill_set"],
        data["correct"]
    ):

        if not skill_set:
            continue

        weight = 1.0 / len(skill_set)

        for skill in skill_set:

            if skill not in skill_entries:
                skill_entries[skill] = []

            skill_entries[skill].append(
                (order_id, weight, weight * correct)
            )

    skill_index = {}

    for skill, entries in skill_entries.items():

        entries = sorted(entries, key=lambda x: x[0])

        orders = np.array(
            [entry[0] for entry in entries]
        )

        weights = np.array(
            [entry[1] for entry in entries],
            dtype=float
        )

        weighted_correct = np.array(
            [entry[2] for entry in entries],
            dtype=float
        )

        skill_index[skill] = {
            "orders": orders,
            "cumulative_weight": np.concatenate(
                [[0.0], np.cumsum(weights)]
            ),
            "cumulative_correct": np.concatenate(
                [[0.0], np.cumsum(weighted_correct)]
            )
        }

    return skill_index


def _query_fractional_skill_index(skill_index, target_skills, order_id):
    """
    Retrieve weighted history from strictly earlier interactions.
    """

    total_weight = 0.0
    total_correct = 0.0

    for skill in target_skills:

        if skill not in skill_index:
            continue

        skill_data = skill_index[skill]

        position = np.searchsorted(
            skill_data["orders"],
            order_id,
            side="left"
        )

        total_weight += skill_data["cumulative_weight"][position]

        total_correct += skill_data["cumulative_correct"][position]

    return total_weight, total_correct


def add_fractional_same_skill_history_features(
    data,
    alpha=10,
    prior_data=None
):
    """
    Generate fractional same-skill historical features.

    Student history:
        Earlier interactions of the same student in `data`.

    Global prior:
        Earlier interactions in `prior_data`.

    Only interactions with order_id strictly smaller than
    the target interaction are considered.
    """

    if prior_data is None:
        prior_data = data

    result = data.copy()

    # Build global chronological fractional history index
    global_skill_index = _build_fractional_skill_index(
        prior_data
    )

    # Build student-specific fractional history indexes
    student_skill_indexes = {}

    for user_id, student_data in data.groupby(
        "user_id",
        sort=False
    ):

        student_skill_indexes[user_id] = (
            _build_fractional_skill_index(student_data)
        )

    weighted_interactions = []
    weighted_correct = []
    weighted_global_prior = []
    weighted_smoothed_accuracy = []

    for user_id, order_id, target_skills in zip(
        result["user_id"],
        result["order_id"],
        result["skill_set"]
    ):

        # Student-specific previous history
        student_index = student_skill_indexes[user_id]

        student_count, student_correct = (
            _query_fractional_skill_index(
                student_index,
                target_skills,
                order_id
            )
        )

        # Chronological global prior
        global_count, global_correct = (
            _query_fractional_skill_index(
                global_skill_index,
                target_skills,
                order_id
            )
        )

        # Global prior with cold-start fallback
        if global_count > 0:
            prior = global_correct / global_count
        else:
            prior = 0.5

        # Smoothed fractional historical accuracy
        smoothed_accuracy = (
            student_correct + alpha * prior
        ) / (
            student_count + alpha
        )

        weighted_interactions.append(student_count)
        weighted_correct.append(student_correct)
        weighted_global_prior.append(prior)
        weighted_smoothed_accuracy.append(smoothed_accuracy)

    result["weighted_same_skill_previous_interactions"] = (
        weighted_interactions
    )

    result["weighted_same_skill_previous_correct"] = (
        weighted_correct
    )

    result["weighted_same_skill_global_prior"] = (
        weighted_global_prior
    )

    result["weighted_same_skill_smoothed_accuracy"] = (
        weighted_smoothed_accuracy
    )

    return result


def _normalise_other_skill_set(value):
    """Convert skill information into a Python set."""
    if isinstance(value, (set, frozenset)):
        return set(value)

    if value is None:
        return set()

    try:
        if pd.isna(value):
            return set()
    except (TypeError, ValueError):
        pass

    return set(str(value).split("_"))


def _build_binary_skill_index(data):
    """
    Map each skill to the positions of questions containing it.
    """
    skill_index = {}

    for position, (order_id, skills) in enumerate(
        zip(data["order_id"], data["skill_set"])
    ):
        for skill in skills:
            skill_index.setdefault(skill, []).append(
                (order_id, position)
            )

    result = {}

    for skill, entries in skill_index.items():
        entries.sort(key=lambda x: x[0])

        result[skill] = (
            np.array([x[0] for x in entries]),
            np.array([x[1] for x in entries], dtype=int)
        )

    return result


def _query_binary_skill_index(skill_index, target_skills, order_id):
    """
    Return unique earlier questions sharing at least one target skill.
    """
    matched_positions = set()

    for skill in target_skills:
        if skill not in skill_index:
            continue

        skill_orders, positions = skill_index[skill]

        cutoff = np.searchsorted(
            skill_orders,
            order_id,
            side="left"
        )

        matched_positions.update(
            positions[:cutoff].tolist()
        )

    return matched_positions


def _build_other_skill_history_index(data):
    """
    Build total, binary-overlap and fractional-overlap indexes.
    """
    total_orders = data["order_id"].to_numpy()
    outcomes = data["correct"].to_numpy()

    eligible_mask = np.array([
        bool(skills) for skills in data["skill_set"]
    ])

    eligible_orders = total_orders[eligible_mask]
    eligible_correct = outcomes[eligible_mask]

    cumulative_correct = np.concatenate(
        [[0], np.cumsum(eligible_correct)]
    )

    binary_index = _build_binary_skill_index(data)

    fractional_index = _build_fractional_skill_index(data)

    return {
        "eligible_orders": eligible_orders,
        "cumulative_correct": cumulative_correct,
        "binary_index": binary_index,
        "fractional_index": fractional_index,
        "outcomes": outcomes
    }


def _query_total_history(index, order_id):
    """Retrieve all eligible earlier interactions."""
    position = np.searchsorted(
        index["eligible_orders"],
        order_id,
        side="left"
    )

    count = position
    correct = index["cumulative_correct"][position]

    return count, correct


def add_other_skill_history_features(
    data,
    alpha=10,
    prior_data=None
):
    """
    Generate binary and fractional other-skill history features.

    Binary other-skill weight:
        1 if previous and target skill sets are disjoint, else 0.

    Fractional other-skill weight:
        Number of previous-question skills outside the target skill set
        divided by the number of skills in the previous question.

    Student history:
        Earlier eligible interactions from the same student in data.

    Global prior:
        Earlier eligible interactions in prior_data.

    Only interactions with strictly smaller order_id are considered.
    """

    if prior_data is None:
        prior_data = data

    # --------------------------------------------------
    # 1. Prepare target and prior datasets
    # --------------------------------------------------

    result = data.copy()

    if "skill_set" in result.columns:
        result["skill_set"] = result["skill_set"].apply(
            _normalise_other_skill_set
        )
    else:
        result["skill_set"] = result["skill_id"].apply(
            _normalise_other_skill_set
        )

    result = result.sort_values(
        ["user_id", "order_id"]
    ).reset_index(drop=True)

    prior_df = prior_data.copy()

    if "skill_set" in prior_df.columns:
        prior_df["skill_set"] = prior_df["skill_set"].apply(
            _normalise_other_skill_set
        )
    else:
        prior_df["skill_set"] = prior_df["skill_id"].apply(
            _normalise_other_skill_set
        )

    prior_df = prior_df.sort_values(
        "order_id"
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 2. Build the chronological global history index
    # --------------------------------------------------

    global_index = _build_other_skill_history_index(
        prior_df
    )

    # --------------------------------------------------
    # 3. Build student-specific history indexes
    # --------------------------------------------------

    student_indexes = {}

    for user_id, student_data in result.groupby(
        "user_id",
        sort=False
    ):
        student_indexes[user_id] = (
            _build_other_skill_history_index(student_data)
        )

    # --------------------------------------------------
    # 4. Calculate historical features
    # --------------------------------------------------

    binary_interactions = []
    binary_correct = []
    binary_global_prior = []

    fractional_interactions = []
    fractional_correct = []
    fractional_global_prior = []

    for user_id, order_id, target_skills in zip(
        result["user_id"],
        result["order_id"],
        result["skill_set"]
    ):

        # If the target has no known skills, its skill relationship
        # with previous questions cannot be determined.
        if not target_skills:
            binary_interactions.append(0.0)
            binary_correct.append(0.0)
            binary_global_prior.append(0.5)

            fractional_interactions.append(0.0)
            fractional_correct.append(0.0)
            fractional_global_prior.append(0.5)

            continue

        # ==============================================
        # A. Student-specific historical information
        # ==============================================

        student_index = student_indexes[user_id]

        total_count, total_correct = _query_total_history(
            student_index,
            order_id
        )

        # Previous questions sharing at least one skill
        # with the target question.
        overlapping_positions = _query_binary_skill_index(
            student_index["binary_index"],
            target_skills,
            order_id
        )

        overlapping_count = len(overlapping_positions)

        overlapping_correct = sum(
            student_index["outcomes"][position]
            for position in overlapping_positions
        )

        # Binary other-skill history
        binary_count = total_count - overlapping_count

        binary_correct_count = (
            total_correct - overlapping_correct
        )

        # Fractional same-skill contribution
        same_weight, same_weighted_correct = (
            _query_fractional_skill_index(
                student_index["fractional_index"],
                target_skills,
                order_id
            )
        )

        # Fractional other-skill history
        fractional_count = total_count - same_weight

        fractional_correct_count = (
            total_correct - same_weighted_correct
        )

        # Avoid tiny negative floating-point values.
        binary_count = max(0.0, binary_count)
        binary_correct_count = max(0.0, binary_correct_count)

        fractional_count = max(0.0, fractional_count)
        fractional_correct_count = max(
            0.0, fractional_correct_count
        )

        # ==============================================
        # B. Chronological global prior
        # ==============================================

        global_total_count, global_total_correct = (
            _query_total_history(
                global_index,
                order_id
            )
        )

        global_overlapping_positions = (
            _query_binary_skill_index(
                global_index["binary_index"],
                target_skills,
                order_id
            )
        )

        global_overlapping_count = len(
            global_overlapping_positions
        )

        global_overlapping_correct = sum(
            global_index["outcomes"][position]
            for position in global_overlapping_positions
        )

        # Binary global other-skill prior
        global_binary_count = (
            global_total_count - global_overlapping_count
        )

        global_binary_correct = (
            global_total_correct - global_overlapping_correct
        )

        # Fractional global other-skill prior
        global_same_weight, global_same_weighted_correct = (
            _query_fractional_skill_index(
                global_index["fractional_index"],
                target_skills,
                order_id
            )
        )

        global_fractional_count = (
            global_total_count - global_same_weight
        )

        global_fractional_correct = (
            global_total_correct - global_same_weighted_correct
        )

        global_binary_count = max(0.0, global_binary_count)
        global_fractional_count = max(
            0.0, global_fractional_count
        )

        if global_binary_count > 0:
            binary_prior = (
                global_binary_correct / global_binary_count
            )
        else:
            binary_prior = 0.5

        if global_fractional_count > 0:
            fractional_prior = (
                global_fractional_correct / global_fractional_count
            )
        else:
            fractional_prior = 0.5

        # ==============================================
        # C. Store results
        # ==============================================

        binary_interactions.append(binary_count)
        binary_correct.append(binary_correct_count)
        binary_global_prior.append(binary_prior)

        fractional_interactions.append(fractional_count)
        fractional_correct.append(fractional_correct_count)
        fractional_global_prior.append(fractional_prior)

    # --------------------------------------------------
    # 5. Store historical counts and global priors
    # --------------------------------------------------

    result["other_skill_previous_interactions"] = (
        binary_interactions
    )

    result["other_skill_previous_correct"] = (
        binary_correct
    )

    result["other_skill_global_prior"] = (
        binary_global_prior
    )

    result["weighted_other_skill_previous_interactions"] = (
        fractional_interactions
    )

    result["weighted_other_skill_previous_correct"] = (
        fractional_correct
    )

    result["weighted_other_skill_global_prior"] = (
        fractional_global_prior
    )

    # --------------------------------------------------
    # 6. Calculate smoothed accuracies
    # --------------------------------------------------

    result["other_skill_smoothed_accuracy"] = (
        result["other_skill_previous_correct"]
        + alpha * result["other_skill_global_prior"]
    ) / (
        result["other_skill_previous_interactions"] + alpha
    )

    result["weighted_other_skill_smoothed_accuracy"] = (
        result["weighted_other_skill_previous_correct"]
        + alpha * result["weighted_other_skill_global_prior"]
    ) / (
        result["weighted_other_skill_previous_interactions"]
        + alpha
    )

    # --------------------------------------------------
    # 7. Log-transformed experience features
    # --------------------------------------------------

    result["log_other_skill_previous_interactions"] = np.log1p(
        result["other_skill_previous_interactions"]
    )

    result["log_weighted_other_skill_previous_interactions"] = (
        np.log1p(
            result["weighted_other_skill_previous_interactions"]
        )
    )

    return result


def add_question_difficulty_features(
    data,
    prior_data=None,
    alpha=10
):
    """
    Add chronological question-difficulty features.

    Uses earlier interactions from other students only.

    Parameters
    ----------
    data : pd.DataFrame
        Target interactions.

    prior_data : pd.DataFrame, optional
        Permitted historical interactions.
        Training targets: training fold.
        Validation targets: training fold.

    alpha : float
        Smoothing strength.

    Returns
    -------
    pd.DataFrame
        Target data with question-difficulty features.
    """

    if prior_data is None:
        prior_data = data

    data = data.copy().reset_index(drop=True)
    prior_data = prior_data.copy()

    # --------------------------------------------------
    # 1. Global chronological prior
    # --------------------------------------------------

    global_history = prior_data.sort_values("order_id")

    global_orders = global_history["order_id"].to_numpy()
    global_correct = global_history["correct"].to_numpy()

    global_cumulative_correct = np.cumsum(global_correct)

    target_orders = data["order_id"].to_numpy()

    global_positions = np.searchsorted(
        global_orders,
        target_orders,
        side="left"
    )

    global_previous_correct = np.zeros(len(data), dtype=float)

    valid = global_positions > 0

    global_previous_correct[valid] = (
        global_cumulative_correct[global_positions[valid] - 1]
    )

    global_previous_count = global_positions.astype(float)

    global_prior = np.divide(
        global_previous_correct,
        global_previous_count,
        out=np.full(len(data), 0.5),
        where=global_previous_count > 0
    )

    # --------------------------------------------------
    # 2. Build chronological question-level indexes
    # --------------------------------------------------

    question_index = {}

    for problem_id, group in prior_data.groupby(
        "problem_id", sort=False
    ):

        group = group.sort_values("order_id")

        orders = group["order_id"].to_numpy()
        correct = group["correct"].to_numpy()

        question_index[problem_id] = (
            orders,
            np.cumsum(correct)
        )

    # --------------------------------------------------
    # 3. Build chronological student-question indexes
    # --------------------------------------------------

    student_question_index = {}

    for (problem_id, user_id), group in prior_data.groupby(
        ["problem_id", "user_id"], sort=False
    ):

        group = group.sort_values("order_id")

        orders = group["order_id"].to_numpy()
        correct = group["correct"].to_numpy()

        student_question_index[(problem_id, user_id)] = (
            orders,
            np.cumsum(correct)
        )

    # --------------------------------------------------
    # 4. Query historical counts for each target
    # --------------------------------------------------

    question_counts = np.zeros(len(data), dtype=float)
    question_correct = np.zeros(len(data), dtype=float)

    target_groups = data.groupby("problem_id", sort=False).indices

    for problem_id, target_positions in target_groups.items():

        target_positions = np.asarray(target_positions)

        current_orders = data.loc[
            target_positions, "order_id"
        ].to_numpy()

        current_users = data.loc[
            target_positions, "user_id"
        ].to_numpy()

        # All previous interactions for this question
        if problem_id in question_index:

            orders, cumulative_correct = question_index[problem_id]

            positions = np.searchsorted(
                orders,
                current_orders,
                side="left"
            )

            total_counts = positions.astype(float)

            total_correct = np.zeros(len(positions), dtype=float)

            valid = positions > 0

            total_correct[valid] = (
                cumulative_correct[positions[valid] - 1]
            )

        else:
            total_counts = np.zeros(len(target_positions))
            total_correct = np.zeros(len(target_positions))

        # Previous interactions by the target students
        self_counts = np.zeros(len(target_positions))
        self_correct = np.zeros(len(target_positions))

        for user_id in pd.unique(current_users):

            local_positions = np.flatnonzero(
                current_users == user_id
            )

            key = (problem_id, user_id)

            if key not in student_question_index:
                continue

            orders, cumulative_correct = student_question_index[key]

            positions = np.searchsorted(
                orders,
                current_orders[local_positions],
                side="left"
            )

            self_counts[local_positions] = positions

            valid = positions > 0

            self_correct_values = np.zeros(
                len(local_positions), dtype=float
            )

            self_correct_values[valid] = (
                cumulative_correct[positions[valid] - 1]
            )

            self_correct[local_positions] = self_correct_values

        # Other students' historical interactions
        question_counts[target_positions] = (
            total_counts - self_counts
        )

        question_correct[target_positions] = (
            total_correct - self_correct
        )

    # --------------------------------------------------
    # 5. Create final features
    # --------------------------------------------------

    data["question_previous_interactions"] = question_counts

    data["question_previous_correct"] = question_correct

    data["question_global_prior"] = global_prior

    data["question_smoothed_accuracy"] = (
        question_correct + alpha * global_prior
    ) / (
        question_counts + alpha
    )

    data["log_question_previous_interactions"] = np.log1p(
        question_counts
    )

    return data
