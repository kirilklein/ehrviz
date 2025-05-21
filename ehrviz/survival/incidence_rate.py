import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from ehrviz.survival.prepare import calculate_cumulative_incidence_groups
from typing import Dict, Optional, Tuple


def plot_cumulative_incidence(
    incidence_data: Dict[str, pd.DataFrame],
    group_colors: Optional[Dict[str, str]] = None,
    group_linestyles: Optional[Dict[str, str]] = None,
    offset_days: int = 0,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    legend_loc: str = "best",
    ci_column: str = "cumulative_incidence",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot cumulative incidence rates for multiple groups.

    Parameters:
    -----------
    incidence_data : Dict[str, pd.DataFrame]
        Dictionary mapping group names to their respective incidence DataFrames.
        Each DataFrame should have 'day' and 'cumulative_incidence' columns.
    group_colors : Dict[str, str], optional
        Dictionary mapping group names to their plot colors.
        If None, default colors will be assigned.
    group_linestyles : Dict[str, str], optional
        Dictionary mapping group names to their plot line styles.
        If None, all groups will use solid lines.
    offset_days : int, default=0
        Number of days offset from index date for the start of follow-up.
    title : str, optional
        Plot title. If None, a default title will be generated.
    xlabel : str, optional
        X-axis label. If None, a default label will be generated.
    ylabel : str, optional
        Y-axis label. If None, "Cumulative Incidence Rate (%)" will be used.
    figsize : Tuple[float, float], default=(10, 6)
        Figure size in inches (width, height).
    save_path : str, optional
        If provided, save the figure to this path.
    ax : plt.Axes, optional
        Axes object to plot on. If None, creates new figure and axes.
    legend_loc : str, default="best"
        Location of the legend on the plot.
    ci_column : str, default="cumulative_incidence"
        Name of the column containing cumulative incidence values.

    Returns:
    --------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects containing the plot.
    """
    # Create plot if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Set default colors if not provided
    if group_colors is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        group_colors = {
            group: default_colors[i % len(default_colors)]
            for i, group in enumerate(incidence_data.keys())
        }

    # Set default line styles if not provided
    if group_linestyles is None:
        group_linestyles = {group: "-" for group in incidence_data.keys()}

    # Plot cumulative incidence for each group
    for group_name, df in incidence_data.items():
        color = group_colors.get(group_name, "black")
        linestyle = group_linestyles.get(group_name, "-")

        # Add number of subjects to legend label
        label = f"{group_name}"

        ax.plot(df["day"], df[ci_column], color=color, linestyle=linestyle, label=label)

    # Add labels and title
    if xlabel is None:
        xlabel = f"Days after index date + {offset_days} days"
    ax.set_xlabel(xlabel)

    if ylabel is None:
        ylabel = "Cumulative Incidence Rate (%)"
    ax.set_ylabel(ylabel)

    if title is None:
        title = f"Cumulative Incidence Rates (Offset: {offset_days} days)"
    ax.set_title(title)

    # Add legend and grid
    ax.legend(loc=legend_loc)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # Ensure x-axis starts at 0
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(0, x_max)

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_cumulative_incidence_simple(
    outcome_df: pd.DataFrame,
    index_date_df: pd.DataFrame,
    exposed_ids: list,
    groups: dict,
    weights: dict,
    follow_up_df: pd.DataFrame,
    offset_days: int = 0,
    max_follow_up_days: int = 365,
    at_risk_time_points: list = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Convenience function that combines data preparation and plotting.

    Parameters are the same as the individual functions.

    Returns:
    --------
    tuple[plt.Figure, plt.Axes]
        The figure and axes objects containing the plot.
    """
    # Prepare the data
    incidence_data = calculate_cumulative_incidence_groups(
        outcome_df=outcome_df,
        index_date_df=index_date_df,
        exposed_ids=exposed_ids,
        groups=groups,
        weights=weights,
        offset_days=offset_days,
        max_follow_up_days=max_follow_up_days,
        follow_up_df=follow_up_df,
    )

    # Create the plot
    return plot_cumulative_incidence(
        incidence_data=incidence_data,
        at_risk_time_points=at_risk_time_points,
        figsize=figsize,
        save_path=save_path,
        ax=ax,
    )
