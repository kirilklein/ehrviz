from typing import Dict, List, Optional

import pandas as pd
import numpy as np


def calculate_cumulative_incidence(
    subject_ids: List,
    outcome_df: pd.DataFrame,
    index_date_df: pd.DataFrame,
    offset_days: int = 0,
    max_follow_up_days: int = 365,
    follow_up_df: Optional[pd.DataFrame] = None,
    weights: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Calculate cumulative incidence rates for a group of subjects on a relative timeline.

    Parameters:
    -----------
    subject_ids : List
        List of subject IDs to include in the analysis.
    outcome_df : pd.DataFrame
        DataFrame containing subject outcomes with columns 'subject_id' and 'time'.
    index_date_df : pd.DataFrame
        DataFrame containing index dates with columns 'subject_id' and 'time'.
    offset_days : int, default=0
        Number of days to offset from the index date for the start of follow-up.
    max_follow_up_days : int, default=365
        Maximum number of days to include in the analysis after offset.
    follow_up_df : pd.DataFrame, optional
        DataFrame with columns 'subject_id' and 'end_date' indicating individual follow-up end dates.
    weights : pd.DataFrame, optional
        DataFrame with columns 'subject_id' and 'weight'.

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - 'day': Days from index date (after offset)
        - 'cumulative_incidence': Cumulative incidence rate for each day
        - 'n_at_risk': Number of subjects at risk for each day
        - 'n_events': Cumulative number of events by each day
    """
    # Filter to only include specified subjects and those with index dates
    valid_subjects = set(subject_ids).intersection(set(index_date_df["subject_id"]))

    if not valid_subjects:
        raise ValueError("No valid subjects found with index dates")

    # Create mapping of subject_id to index_date
    index_dates = dict(zip(index_date_df["subject_id"], index_date_df["time"]))

    # Filter outcome data to only include valid subjects
    valid_outcomes = outcome_df[outcome_df["subject_id"].isin(valid_subjects)].copy()

    # Calculate days from index date to outcome for each subject
    valid_outcomes["index_date"] = valid_outcomes["subject_id"].map(index_dates)
    valid_outcomes["days_to_outcome"] = (
        valid_outcomes["time"] - valid_outcomes["index_date"]
    ).dt.days - offset_days

    # Only include outcomes that occurred after the offset
    valid_outcomes = valid_outcomes[valid_outcomes["days_to_outcome"] >= 0]

    # Process individual follow-up times
    follow_up_times = {}
    for subject_id in valid_subjects:
        # Default max follow-up time
        follow_up_times[subject_id] = max_follow_up_days

    if follow_up_df is not None:
        # Convert follow_up_df to a dictionary for faster lookups
        follow_up_dict = follow_up_df.set_index("subject_id")["end_date"].to_dict()

        # Update follow-up times based on individual end dates
        for subject_id in valid_subjects:
            if subject_id in follow_up_dict:
                days_to_end = (
                    follow_up_dict[subject_id] - index_dates[subject_id]
                ).days - offset_days
                if days_to_end >= 0:  # Only consider valid follow-up times
                    follow_up_times[subject_id] = min(
                        follow_up_times[subject_id], days_to_end
                    )

    # Determine the maximum day for analysis
    actual_max_days = min(
        max_follow_up_days,
        max(follow_up_times.values()) if follow_up_times else max_follow_up_days,
    )
    if valid_outcomes.empty:
        actual_max_days = max_follow_up_days
    else:
        valid_outcome_max = valid_outcomes["days_to_outcome"].max()
        if not np.isnan(valid_outcome_max):
            actual_max_days = min(actual_max_days, int(valid_outcome_max) + 1)

    days_range = list(range(0, int(actual_max_days) + 1))

    # Set up result containers
    cumulative_incidence = []
    n_at_risk = []
    n_events = []

    # Use weights if provided, otherwise default to 1.0
    if weights is None:
        weights = pd.DataFrame(
            {"subject_id": list(valid_subjects), "weight": [1.0] * len(valid_subjects)}
        ).set_index("subject_id")
    else:
        weights = weights.set_index("subject_id")
    # Get the earliest outcome day for each subject (if any)
    subject_outcome_days = {}
    for _, row in valid_outcomes.iterrows():
        subject_id = row["subject_id"]
        day = row["days_to_outcome"]
        if (
            subject_id not in subject_outcome_days
            or day < subject_outcome_days[subject_id]
        ):
            subject_outcome_days[subject_id] = day

    # Calculate cumulative incidence for each day
    for day in days_range:
        # Determine subjects at risk on this day
        at_risk_subjects = [
            subject_id
            for subject_id in valid_subjects
            if follow_up_times[subject_id] >= day
            and (
                subject_id not in subject_outcome_days
                or subject_outcome_days[subject_id] > day
            )
        ]

        # Calculate weighted number at risk
        weighted_at_risk = sum(
            weights.loc[subject_id, "weight"] for subject_id in at_risk_subjects
        )
        n_at_risk.append(weighted_at_risk)

        # Determine subjects with events by this day
        event_subjects = [
            subject_id
            for subject_id in valid_subjects
            if subject_id in subject_outcome_days
            and subject_outcome_days[subject_id] <= day
        ]

        # Calculate weighted number of events
        weighted_events = sum(
            weights.loc[subject_id, "weight"] for subject_id in event_subjects
        )
        n_events.append(weighted_events)

        # Calculate cumulative incidence
        total_weighted_subjects = sum(
            weights.loc[subject_id, "weight"] for subject_id in valid_subjects
        )
        if total_weighted_subjects > 0:
            cumulative_incidence.append(weighted_events / total_weighted_subjects)
        else:
            cumulative_incidence.append(0.0)

    # Create result DataFrame
    result_df = pd.DataFrame(
        {
            "day": days_range,
            "cumulative_incidence": cumulative_incidence,
            "n_at_risk": n_at_risk,
            "n_events": n_events,
        }
    )

    return result_df


def calculate_cumulative_incidence_groups(
    group_definitions: Dict[str, List],
    outcome_df: pd.DataFrame,
    index_date_df: pd.DataFrame,
    offset_days: int = 0,
    max_follow_up_days: int = 365,
    follow_up_df: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate and compare cumulative incidence across multiple groups.

    Parameters:
    -----------
    group_definitions : Dict[str, List]
        Dictionary mapping group names to lists of subject IDs.
    outcome_df : pd.DataFrame
        DataFrame containing subject outcomes with columns 'subject_id' and 'time'.
    index_date_df : pd.DataFrame
        DataFrame containing index dates with columns 'subject_id' and 'time'.
    offset_days : int, default=0
        Number of days to offset from the index date for the start of follow-up.
    max_follow_up_days : int, default=365
        Maximum number of days to include in the analysis after offset.
    follow_up_df : pd.DataFrame, optional
        DataFrame with columns 'subject_id' and 'end_date' indicating individual follow-up end dates.
    weights : Dict[Union[str, int], float], optional
        Dictionary mapping subject_ids to their respective weights.

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary mapping group names to their respective incidence DataFrames.
    """
    results = {}

    for group_name, subject_ids in group_definitions.items():
        results[group_name] = calculate_cumulative_incidence(
            subject_ids=subject_ids,
            outcome_df=outcome_df,
            index_date_df=index_date_df,
            offset_days=offset_days,
            max_follow_up_days=max_follow_up_days,
            follow_up_df=follow_up_df,
            weights=weights.get(group_name, None)
            if isinstance(weights, dict)
            else weights,
        )

    return results
