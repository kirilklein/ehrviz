import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from ehrviz.survival.prepare import prepare_incidence_data
from ehrviz.survival.helper import compute_risk_table, add_risk_table_to_plot


def plot_cumulative_incidence(
    incidence_data: dict,
    at_risk_time_points: list = None,
    show_risk_table: bool = True,
    figsize: tuple = (10, 6),
    save_path: str = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot cumulative incidence rates based on the processed data.

    Args:
        incidence_data (dict): Dictionary containing the processed data
        at_risk_time_points (list, optional): Time points to show in the at-risk table.
            If None, will use [0, 60, 120, 180, 240, 300]. Defaults to None.
        show_risk_table (bool, optional): Whether to show the risk table. Defaults to True.
        figsize (tuple, optional): Figure size in inches (width, height). Defaults to (10, 6).
        save_path (str, optional): If provided, save the figure to this path. Defaults to None.
        ax (plt.Axes, optional): Axes object to plot on. If None, creates new figure and axes.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects containing the plot.
    """
    # Extract data from the dictionary
    days = incidence_data["days"]
    exposed_incidence = incidence_data["exposed_incidence"]
    unexposed_incidence = incidence_data["unexposed_incidence"]
    total_exposed = incidence_data["total_exposed"]
    total_unexposed = incidence_data["total_unexposed"]
    valid_outcomes = incidence_data["valid_outcomes"]
    offset_days = incidence_data["offset_days"]

    # Set default at-risk time points if not provided
    if at_risk_time_points is None:
        at_risk_time_points = [0, 60, 120, 180, 240, 300]

    max_days = max(days)

    # Create plot if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot the incidence curves
    ax.plot(days, exposed_incidence, "r-", label="Exposed")
    ax.plot(days, unexposed_incidence, "b-", label="Unexposed")

    # Add labels and title
    ax.set_xlabel(f"Days after index date + {offset_days} days")
    ax.set_ylabel("Cumulative Incidence Rate (%)")
    ax.set_title(f"Cumulative Incidence Rates (Offset: {offset_days} days)")

    # Add legend and grid
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # Add risk table if requested
    if show_risk_table:
        risk_table_data = compute_risk_table(
            valid_outcomes=valid_outcomes,
            total_exposed=total_exposed,
            total_unexposed=total_unexposed,
            time_points=at_risk_time_points,
            max_days=max_days,
        )
        add_risk_table_to_plot(ax, risk_table_data, fig)

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_cumulative_incidence_simple(
    outcome_df: pd.DataFrame,
    index_date_df: pd.DataFrame,
    exposed_ids: list,
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
    incidence_data = prepare_incidence_data(
        outcome_df=outcome_df,
        index_date_df=index_date_df,
        exposed_ids=exposed_ids,
        offset_days=offset_days,
        max_follow_up_days=max_follow_up_days,
    )

    # Create the plot
    return plot_cumulative_incidence(
        incidence_data=incidence_data,
        at_risk_time_points=at_risk_time_points,
        figsize=figsize,
        save_path=save_path,
        ax=ax,
    )
