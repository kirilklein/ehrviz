import pandas as pd

from ehrviz.constants.data import (
    SEGMENT_DURATION,
    SEGMENT_END,
    SEGMENT_START,
    SUBJECT_ID,
    TIME,
)


def compute_continuous_treatment_durations(
    df, subject_col=SUBJECT_ID, time_col=TIME, interruption_days=30
):
    """
    Compute continuous treatment segments for each patient in a vectorized manner.

    A continuous treatment segment is defined as successive treatment events where
    the gap between events is less than interruption_days. When a gap is at least
    interruption_days, a new segment is started.

    Parameters:
      df (pd.DataFrame): DataFrame with treatment event data.
      subject_col (str): Column name for patient identifiers.
      time_col (str): Column name for event timestamps.
      interruption_days (int or float): Threshold (in days) to define a break between segments.

    Returns:
      pd.DataFrame: A DataFrame with columns: subject_col, segment_start, segment_end, duration,
                    where duration is computed in days.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df.sort_values([subject_col, time_col], inplace=True)

    # Flag when a new segment starts (either a new subject or a gap ≥ interruption_days)
    new_segment = (df[subject_col] != df[subject_col].shift()) | (
        (df[time_col] - df[time_col].shift()) >= pd.Timedelta(days=interruption_days)
    )
    df["segment_id"] = new_segment.cumsum()

    segments = (
        df.groupby(["segment_id", subject_col])[time_col]
        .agg(["first", "last"])
        .reset_index()
    )
    segments.rename(columns={"first": SEGMENT_START, "last": SEGMENT_END}, inplace=True)
    segments["duration"] = (
        segments["segment_end"] - segments["segment_start"]
    ).dt.total_seconds() / (3600 * 24)

    return segments[[subject_col, SEGMENT_START, SEGMENT_END, SEGMENT_DURATION]]


def compute_total_treatment_duration(df, subject_col=SUBJECT_ID, time_col=TIME):
    """
    Compute total treatment duration for each subject.

    Total duration is defined as the difference (in days) between the last and the first treatment events.

    Parameters:
      df (pd.DataFrame): DataFrame with treatment event data.
      subject_col (str): Column name for patient identifiers.
      time_col (str): Column name for event timestamps.

    Returns:
      pd.DataFrame: A DataFrame with columns: subject_col and total_duration (in days).
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    grouped = df.groupby(subject_col)[time_col].agg(["min", "max"]).reset_index()
    grouped["total_duration"] = (grouped["max"] - grouped["min"]).dt.total_seconds() / (
        3600 * 24
    )
    return grouped[[subject_col, "total_duration"]]
