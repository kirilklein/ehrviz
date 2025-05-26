from typing import List, Optional

import numpy as np
import pandas as pd


def prepare_survival_data_for_lifelines(
    index_dates: pd.DataFrame,
    outcomes: pd.DataFrame,
    censor_dates: pd.DataFrame,
    offset_days: int = 0,
    max_follow_up_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create a survival analysis dataframe in lifelines format.

    This function creates a compact representation with one row per subject,
    containing duration (T) and event_observed (E) columns as expected by lifelines.

    Parameters:
    -----------
    index_dates : pd.DataFrame
        DataFrame with columns ['subject_id', 'time'] - determines patient cohort and start dates
    outcomes : pd.DataFrame
        DataFrame with columns ['subject_id', 'time'] - outcome timestamps
    censor_dates : pd.DataFrame
        DataFrame with columns ['subject_id', 'time'] - censoring dates for each subject
    offset_days : int, default=0
        Number of days to offset from index_date (observation window start)
    max_follow_up_days : int, optional
        Maximum follow-up period in days

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: subject_id, T (duration), E (event_observed)
        - T: Time to event or censoring (in days)
        - E: 1 if event occurred, 0 if censored
    """

    _ensure_datetime_columns([index_dates, outcomes, censor_dates])

    subjects_df = _prepare_subjects_baseline(index_dates, censor_dates, offset_days)

    subjects_df = _find_first_outcomes(outcomes, subjects_df)

    subjects_df = _determine_events(subjects_df, max_follow_up_days)

    # Create lifelines format: one row per subject with T and E
    lifelines_df = subjects_df[["subject_id", "event_day", "is_outcome"]].copy()
    lifelines_df = lifelines_df.rename(columns={"event_day": "T", "is_outcome": "E"})

    # Convert boolean to int for E column
    lifelines_df["E"] = lifelines_df["E"].astype(int)
    lifelines_df["T"] = lifelines_df["T"].astype(int)

    return lifelines_df[["subject_id", "T", "E"]]


def _ensure_datetime_columns(dataframes: List[pd.DataFrame]) -> None:
    """Ensure 'time' columns are datetime type."""
    for df in dataframes:
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])


def _prepare_subjects_baseline(
    index_dates: pd.DataFrame, censor_dates: pd.DataFrame, offset_days: int
) -> pd.DataFrame:
    """Merge index and censor dates, calculate observation windows."""
    subjects_df = index_dates.merge(
        censor_dates, on="subject_id", suffixes=("_index", "_censor"), how="inner"
    )

    subjects_df["observation_start"] = subjects_df["time_index"] + pd.Timedelta(
        days=offset_days
    )
    subjects_df["censor_day"] = (
        subjects_df["time_censor"] - subjects_df["observation_start"]
    ).dt.days.clip(lower=0)

    return subjects_df


def _find_first_outcomes(
    outcomes: pd.DataFrame, subjects_df: pd.DataFrame
) -> pd.DataFrame:
    """Find first outcome after observation start for each subject."""
    if len(outcomes) == 0:
        subjects_df["first_outcome_time"] = pd.NaT
        return subjects_df

    outcomes_with_obs_start = outcomes.merge(
        subjects_df[["subject_id", "observation_start"]], on="subject_id", how="inner"
    )

    valid_outcomes = outcomes_with_obs_start[
        outcomes_with_obs_start["time"] >= outcomes_with_obs_start["observation_start"]
    ]

    if len(valid_outcomes) > 0:
        first_outcomes = (
            valid_outcomes.groupby("subject_id")["time"].min().reset_index()
        )
        first_outcomes.columns = ["subject_id", "first_outcome_time"]
        subjects_df = subjects_df.merge(first_outcomes, on="subject_id", how="left")
    else:
        subjects_df["first_outcome_time"] = pd.NaT

    return subjects_df


def _determine_events(
    subjects_df: pd.DataFrame, max_follow_up_days: Optional[int]
) -> pd.DataFrame:
    """Determine event day and type (outcome vs censoring)."""
    subjects_df["outcome_day"] = (
        subjects_df["first_outcome_time"] - subjects_df["observation_start"]
    ).dt.days.clip(lower=0)

    has_outcome = ~subjects_df["first_outcome_time"].isna()
    outcome_before_censor = has_outcome & (
        subjects_df["first_outcome_time"] <= subjects_df["time_censor"]
    )

    subjects_df["event_day"] = np.where(
        outcome_before_censor, subjects_df["outcome_day"], subjects_df["censor_day"]
    )
    subjects_df["is_outcome"] = outcome_before_censor

    if max_follow_up_days is not None:
        subjects_df["event_day"] = subjects_df["event_day"].clip(
            upper=max_follow_up_days
        )

    return subjects_df[subjects_df["event_day"] >= 0].copy()
