from ehrviz.constants.data import (
    SUBJECT_ID,
    SEGMENT_START,
    SEGMENT_END,
    SEGMENT_DURATION,
)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_treatment_duration_distribution_from_segments(
    segments_df,
    subject_col=SUBJECT_ID,
    start_col=SEGMENT_START,
    end_col=SEGMENT_END,
    duration_col=SEGMENT_DURATION,
    bins=30,
    ax=None,
    title_total_duration="Treatment Duration Distribution",
    title_longest_duration="Longest Continuous Treatment Duration",
    xlabel_total_duration="Total Duration (days)",
    ylabel_total_duration="Number of Patients",
    xlabel_longest_duration="Longest Continuous Duration (days)",
    ylabel_longest_duration="Number of Patients",
    **hist_kwargs,
):
    """
    Given precomputed treatment segments, plot two duration metrics:
      1. Total treatment duration per subject (from the earliest start to the latest end).
      2. Longest continuous treatment duration per subject (maximum segment duration).

    Parameters:
      segments_df (pd.DataFrame): DataFrame with one row per continuous treatment segment.
          Expected columns: subject_col, start_col, end_col, duration_col.
      subject_col (str): Column name for patient identifiers.
      start_col (str): Column name for the segment start time.
      end_col (str): Column name for the segment end time.
      duration_col (str): Column name for the segment duration (in days).
      bins (int): Number of bins for the histograms.
      ax (array-like of matplotlib.axes.Axes, optional): If provided, should be 1x2 axes for plotting.
      hist_kwargs: Additional keyword arguments forwarded to plt.hist().

    Returns:
      matplotlib.axes.Axes: The axes on which the histograms were drawn.
    """
    segments_df = segments_df.copy()
    segments_df[start_col] = pd.to_datetime(segments_df[start_col])
    segments_df[end_col] = pd.to_datetime(segments_df[end_col])

    # Total duration per subject: from the minimum start to the maximum end.
    total_durations = segments_df.groupby(subject_col).agg(
        {start_col: "min", end_col: "max"}
    )
    total_durations["total_duration"] = (
        total_durations[end_col] - total_durations[start_col]
    ).dt.total_seconds() / (3600 * 24)

    # Longest continuous duration per subject.
    longest_continuous = segments_df.groupby(subject_col)[duration_col].max()

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    elif not hasattr(ax, "__iter__"):
        ax = [ax]

    ax[0].hist(
        total_durations["total_duration"],
        bins=bins,
        color="skyblue",
        edgecolor="black",
        **hist_kwargs,
    )
    ax[0].set_xlabel("Total Duration (days)")
    ax[0].set_ylabel("Number of Patients")
    ax[0].set_title(title_total_duration)

    ax[1].hist(
        longest_continuous, bins=bins, color="salmon", edgecolor="black", **hist_kwargs
    )
    ax[1].set_xlabel(xlabel_longest_duration)
    ax[1].set_ylabel(ylabel_longest_duration)
    ax[1].set_title(title_longest_duration)

    return ax


def set_monthly_ticks_with_labels_every(ax, label_interval=3, tick_interval=1):
    """
    Set x-axis ticks so that ticks are displayed for every month, but labels are shown only every
    label_interval months.

    Parameters:
      ax (matplotlib.axes.Axes): The axes to customize.
      label_interval (int): Interval in months for labeling ticks.
      tick_interval (int): Interval in months at which ticks appear.
    """
    # Show a tick every 'tick_interval' month (as minor ticks).
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=tick_interval))
    # Show major ticks (with labels) every 'label_interval' months.
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=label_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def plot_currently_treated_patients(
    segments_df,
    start_col=SEGMENT_START,
    end_col=SEGMENT_END,
    freq="W",
    labels_every_n_months=3,
    ticks_every_n_months=1,
    title="Currently Treated Patients Over Time",
    ax=None,
    line_kwargs=None,
    rotation=45,
):
    """
    Plot the number of currently treated patients over time using precomputed treatment segments.

    A patient is considered "currently treated" at time t if t falls between a segment's start and end.
    Time is binned using the frequency provided (default 'W' for weekly). The x-axis will display
    monthly ticks, but with labels only every labels_every_n_months.

    Parameters:
      segments_df (pd.DataFrame): DataFrame with treatment segments. Expected columns include
          'segment_start' and 'segment_end'.
      subject_col (str): Column name for patient identifiers.
      freq (str): Frequency for time bins (e.g., 'W' for weekly).
      labels_every_n_months (int): Interval (in months) at which to show axis labels.
      ticks_every_n_months (int): Interval (in months) at which to place ticks.
      title (str): Plot title.
      ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, a new figure and axis are created.
      line_kwargs (dict, optional): Additional keyword arguments passed to ax.plot().
      rotation (int or float): Rotation angle for x-axis labels.

    Returns:
      matplotlib.axes.Axes: The axis on which the plot was drawn.
    """
    segments_df = segments_df.copy()
    segments_df[start_col] = pd.to_datetime(segments_df[start_col])
    segments_df[end_col] = pd.to_datetime(segments_df[end_col])

    global_start = segments_df[start_col].min()
    global_end = segments_df[end_col].max()

    t_bins = pd.date_range(start=global_start, end=global_end, freq=freq)
    counts = [
        ((segments_df[start_col] <= t) & (segments_df[end_col] >= t)).sum()
        for t in t_bins
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    if line_kwargs is None:
        line_kwargs = {}

    ax.plot(t_bins, counts, marker="o", linestyle="-", **line_kwargs)
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Currently Treated Patients")
    ax.set_title(title)

    set_monthly_ticks_with_labels_every(
        ax, label_interval=labels_every_n_months, tick_interval=ticks_every_n_months
    )
    plt.setp(ax.get_xticklabels(), rotation=rotation)
    return ax
