import pandas as pd


def prepare_incidence_data(
    outcome_df: pd.DataFrame,
    index_date_df: pd.DataFrame,
    exposed_ids: list,
    offset_days: int = 0,
    max_follow_up_days: int = 365,
) -> dict:
    """
    Prepare cumulative incidence data for exposed and unexposed subjects on a relative timeline.

    Parameters:
    -----------
    outcome_df : pandas.DataFrame
        DataFrame containing subject outcomes with columns 'subject_id' and 'time'.
    index_date_df : pandas.DataFrame
        DataFrame containing index dates with columns 'subject_id' and 'time'.
    exposed_ids : list
        List of subject IDs that are considered exposed.
    offset_days : int, default=0
        Number of days to offset from the index date for the start of follow-up.
    max_follow_up_days : int, default=365
        Maximum number of days to include in the analysis after offset.

    Returns:
    --------
    dict
        Dictionary containing the processed data:
        - days: List of days for the x-axis
        - exposed_incidence: List of cumulative incidence values for exposed group
        - unexposed_incidence: List of cumulative incidence values for unexposed group
        - total_exposed: Total number of exposed subjects
        - total_unexposed: Total number of unexposed subjects
        - valid_outcomes: DataFrame of valid outcomes with additional columns
    """
    # Create a dictionary to map subject_id to their exposure status
    exposure_status = {
        subject_id: "Exposed" if subject_id in exposed_ids else "Unexposed"
        for subject_id in index_date_df["subject_id"]
    }

    # Add exposure status to index_date_df
    index_date_df = index_date_df.copy()
    index_date_df["exposure_status"] = index_date_df["subject_id"].map(exposure_status)

    # Create a dictionary mapping subject_id to index_date
    index_dates = dict(zip(index_date_df["subject_id"], index_date_df["time"]))

    # Only include subjects who have an index date
    valid_outcomes = outcome_df[
        outcome_df["subject_id"].isin(index_date_df["subject_id"])
    ].copy()

    if len(valid_outcomes) == 0:
        raise ValueError("No outcomes found for subjects with index dates")

    # Calculate time from index date (plus offset) to outcome for each subject
    valid_outcomes["index_date"] = valid_outcomes["subject_id"].map(index_dates)
    valid_outcomes["days_to_outcome"] = (
        valid_outcomes["time"] - valid_outcomes["index_date"]
    ).dt.days - offset_days

    # Only include outcomes that occurred after the offset
    valid_outcomes = valid_outcomes[valid_outcomes["days_to_outcome"] >= 0]

    # Add exposure status to outcomes
    valid_outcomes["exposure_status"] = valid_outcomes["subject_id"].map(
        exposure_status
    )

    # Count total subjects in each group
    total_exposed = sum(
        1 for subject_id in index_date_df["subject_id"] if subject_id in exposed_ids
    )
    total_unexposed = len(index_date_df) - total_exposed

    # Set the maximum follow-up duration
    max_days = (
        min(max_follow_up_days, valid_outcomes["days_to_outcome"].max() + 30)
        if len(valid_outcomes) > 0
        else max_follow_up_days
    )
    days = list(range(0, int(max_days) + 1))

    # Calculate cumulative incidence for each group
    exposed_incidence = []
    unexposed_incidence = []

    for day in days:
        # For exposed group
        outcomes_by_day_exp = valid_outcomes[
            (valid_outcomes["exposure_status"] == "Exposed")
            & (valid_outcomes["days_to_outcome"] <= day)
        ]
        exposed_count = len(outcomes_by_day_exp["subject_id"].unique())
        exposed_incidence.append(
            exposed_count / total_exposed if total_exposed > 0 else 0
        )

        # For unexposed group
        outcomes_by_day_unexp = valid_outcomes[
            (valid_outcomes["exposure_status"] == "Unexposed")
            & (valid_outcomes["days_to_outcome"] <= day)
        ]
        unexposed_count = len(outcomes_by_day_unexp["subject_id"].unique())
        unexposed_incidence.append(
            unexposed_count / total_unexposed if total_unexposed > 0 else 0
        )

    # Return the processed data
    return {
        "days": days,
        "exposed_incidence": exposed_incidence,
        "unexposed_incidence": unexposed_incidence,
        "total_exposed": total_exposed,
        "total_unexposed": total_unexposed,
        "valid_outcomes": valid_outcomes,
        "offset_days": offset_days,
    }
