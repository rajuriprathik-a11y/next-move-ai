
import pandas as pd


def load_cv_data(project_root, chunksize=50000):
    """
    Load training-pool interactions and reconstruct
    the five student-grouped cross-validation folds.

    Parameters
    ----------
    project_root : pathlib.Path
        Root directory of the project.

    chunksize : int
        Number of CSV rows read at a time.

    Returns
    -------
    fold_datasets : list
        Five tuples of (fold_train_df, fold_val_df).
    """

    # 1. Define file paths
    split_path = (
        project_root / "data" / "processed" / "split_assignments.csv"
    )

    data_path = (
        project_root / "data" / "processed" / "main_questions_cleaned.csv"
    )

    # 2. Load student assignments
    split_assignments = pd.read_csv(split_path)

    train_assignments = split_assignments[
        split_assignments["partition"] == "train_pool"
    ].copy()

    train_assignments["validation_fold"] = (
        train_assignments["validation_fold"].astype(int)
    )

    train_student_ids = set(train_assignments["user_id"])

    # 3. Load only training-pool interactions
    train_chunks = []

    for chunk in pd.read_csv(data_path, chunksize=chunksize):

        training_rows = chunk[
            chunk["user_id"].isin(train_student_ids)
        ]

        if not training_rows.empty:
            train_chunks.append(training_rows)

    train_df = pd.concat(train_chunks, ignore_index=True)

    # 4. Sort interactions chronologically
    train_df = (
        train_df.sort_values(["user_id", "order_id"])
                .reset_index(drop=True)
    )

    # 5. Reconstruct the five CV folds
    fold_datasets = []

    for fold_number in range(1, 6):

        val_student_ids = train_assignments.loc[
            train_assignments["validation_fold"] == fold_number,
            "user_id"
        ]

        fold_train_student_ids = train_assignments.loc[
            train_assignments["validation_fold"] != fold_number,
            "user_id"
        ]

        fold_train_df = train_df[
            train_df["user_id"].isin(fold_train_student_ids)
        ].copy()

        fold_val_df = train_df[
            train_df["user_id"].isin(val_student_ids)
        ].copy()

        # 6. Verify student separation
        overlap = (
            set(fold_train_df["user_id"])
            & set(fold_val_df["user_id"])
        )

        if overlap:
            raise ValueError(
                f"Student leakage detected in fold {fold_number}"
            )

        fold_datasets.append(
            (fold_train_df, fold_val_df)
        )

    return fold_datasets
