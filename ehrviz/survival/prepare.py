from typing import List, Optional

import numpy as np
import pandas as pd


def prepare_survival_data(
    index_dates: pd.DataFrame,
    outcomes: pd.DataFrame,
    censor_dates: pd.DataFrame,
    offset_days: int = 0,
    max_follow_up_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create a survival analysis dataframe from multiple input dataframes.

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
        Processed dataframe with columns: subject_id, day, censored, outcome
    """

    _ensure_datetime_columns([index_dates, outcomes, censor_dates])

    subjects_df = _prepare_subjects_baseline(index_dates, censor_dates, offset_days)

    subjects_df = _find_first_outcomes(outcomes, subjects_df)

    subjects_df = _determine_events(subjects_df, max_follow_up_days)

    return _create_daily_records(subjects_df)


def calculate_cumulative_incidence(
    data: pd.DataFrame,
    subject_list: list = None,
    weights: dict = None,
    method: str = "kaplan_meier",
) -> pd.DataFrame:
    """
    Calculate cumulative incidence rate using Kaplan-Meier-like approach or crude method.

    Parameters:
    -----------
    data : pd.DataFrame, optional
        Survival data (dataframe with columns subject_id, day, censored, outcome)
    subject_list : List, optional
        List of subject_ids to include in analysis
    weights : Dict, optional
        Dictionary mapping subject_id to weight for weighted analysis
    method : str, default="kaplan_meier"
        Method for calculating cumulative incidence:
        - "kaplan_meier": Adjusts denominator as subjects leave risk set (default)
        - "crude": Uses initial population as constant denominator

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: day, at_risk, events, cumulative_incidence
    """
    if data.empty:
        return pd.DataFrame(
            columns=[
                "day",
                "at_risk",
                "events",
                "subjects_at_risk",
                "hazard",
                "survival",
                "cumulative_incidence",
            ]
        )

    if method not in ["kaplan_meier", "crude"]:
        raise ValueError("method must be either 'kaplan_meier' or 'crude'")

    data_filtered = _filter_subjects(data, subject_list)
    w = _initialize_weights(data_filtered, weights)
    daily = _compute_daily_counts(data_filtered, w)

    if method == "kaplan_meier":
        result = _add_survival_and_cumulative_km(daily)
    else:  # crude
        result = _add_survival_and_cumulative_crude(daily)

    return result


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


def _create_daily_records(subjects_df: pd.DataFrame) -> pd.DataFrame:
    """Create daily records from event data."""
    if len(subjects_df) == 0:
        return pd.DataFrame(columns=["subject_id", "day", "censored", "outcome"])

    subjects_df["days_range"] = subjects_df["event_day"].apply(
        lambda x: list(range(int(x) + 1))
    )
    daily_df = subjects_df.explode("days_range").reset_index(drop=True)
    daily_df["day"] = daily_df["days_range"].astype(int)

    daily_df["outcome"] = (
        (daily_df["day"] == daily_df["event_day"]) & daily_df["is_outcome"]
    ).astype(int)

    daily_df["censored"] = (
        (daily_df["day"] == daily_df["event_day"]) & ~daily_df["is_outcome"]
    ).astype(int)

    return daily_df[["subject_id", "day", "censored", "outcome"]]


def _filter_subjects(data: pd.DataFrame, subject_list: list = None) -> pd.DataFrame:
    """
    Subset data to only include specified subjects.
    """
    if subject_list is None:
        return data.copy()
    return data[data["subject_id"].isin(subject_list)].copy()


def _initialize_weights(data: pd.DataFrame, weights: dict = None) -> dict:
    """
    Ensure a weights dict; default to 1.0 for each subject.
    """
    if weights is not None:
        return weights
    return {sid: 1.0 for sid in data["subject_id"].unique()}


def _compute_daily_counts(data: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    For each day, compute weighted at-risk and weighted events.
    Subjects are removed from at-risk if they have an outcome or are censored on that day.

    Optimized version using pre-computed lookups.

    Returns a DataFrame with columns:
      - day
      - at_risk
      - events
      - subjects_at_risk
    """
    if data.empty:
        return pd.DataFrame(columns=["day", "at_risk", "events", "subjects_at_risk"])

    max_day = int(data["day"].max())

    # Pre-compute weighted events by day
    data_with_weights = data.copy()
    data_with_weights["weight"] = (
        data_with_weights["subject_id"].map(weights).fillna(1.0)
    )

    events_by_day = (
        data_with_weights[data_with_weights["outcome"] == 1]
        .groupby("day")["weight"]
        .sum()
        .reindex(range(max_day + 1), fill_value=0.0)
        .to_dict()
    )

    # Pre-compute sets of subjects with events by day
    subjects_with_events_by_day = {}
    event_data = data_with_weights[
        (data_with_weights["outcome"] == 1) | (data_with_weights["censored"] == 1)
    ]
    if not event_data.empty:
        subjects_with_events_by_day = (
            event_data.groupby("day")["subject_id"].apply(set).to_dict()
        )

    # Pre-compute max day for each subject
    subject_max_days = data.groupby("subject_id")["day"].max().to_dict()

    records = []
    for day in range(max_day + 1):
        # Get subjects available on this day (have data >= day)
        available_subjects = {
            subj
            for subj, max_day_subj in subject_max_days.items()
            if max_day_subj >= day
        }

        # Remove subjects with events on this day
        subjects_with_events_today = subjects_with_events_by_day.get(day, set())
        at_risk_subjects = available_subjects - subjects_with_events_today

        # Calculate weighted at-risk
        weighted_at_risk = sum(weights.get(sid, 1.0) for sid in at_risk_subjects)

        records.append(
            {
                "day": day,
                "at_risk": weighted_at_risk,
                "events": events_by_day.get(day, 0.0),
                "subjects_at_risk": len(at_risk_subjects),
            }
        )

    return pd.DataFrame(records)


def _add_survival_and_cumulative_km(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given daily counts, compute hazard, survival, and cumulative incidence (%) using Kaplan-Meier method.
    Denominator adjusts as subjects leave the risk set.
    """
    df = df.copy()
    df["hazard"] = np.where(df["at_risk"] > 0, df["events"] / df["at_risk"], 0)
    df["survival"] = (1 - df["hazard"]).cumprod()
    df["cumulative_incidence"] = (1 - df["survival"]) * 100
    return df


def _add_survival_and_cumulative_crude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given daily counts, compute crude cumulative incidence (%).
    Uses initial population as constant denominator.
    """
    df = df.copy()

    # Get initial population size (day 0 at-risk)
    if len(df) == 0:
        return df

    initial_at_risk = df.iloc[0]["at_risk"]

    # Calculate crude hazard using initial population as denominator
    df["hazard"] = np.where(initial_at_risk > 0, df["events"] / initial_at_risk, 0)

    # Crude cumulative incidence is just cumulative sum of daily hazards
    df["cumulative_incidence"] = df["hazard"].cumsum() * 100

    # For crude method, survival is 1 - cumulative_incidence/100
    df["survival"] = 1 - (df["cumulative_incidence"] / 100)

    return df
