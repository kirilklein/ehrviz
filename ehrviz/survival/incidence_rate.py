import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from datetime import datetime, timedelta
from ehrviz.survival.prepare import prepare_incidence_data
from ehrviz.survival.helper import compute_risk_table, add_risk_table_to_plot


def plot_cumulative_incidence(
    incidence_data: dict,
    at_risk_time_points: list = None,
    show_risk_table: bool = True,
    figsize: tuple = (10, 6),
    save_path: str = None,
) -> plt.Figure:
    """
    Plot cumulative incidence rates based on the processed data.

    Args:
        incidence_data (dict): Dictionary containing the processed data
        at_risk_time_points (list, optional): Time points to show in the at-risk table.
            If None, will use [0, 60, 120, 180, 240, 300]. Defaults to None.
        show_risk_table (bool, optional): Whether to show the risk table. Defaults to True.
        figsize (tuple, optional): Figure size in inches (width, height). Defaults to (10, 6).
        save_path (str, optional): If provided, save the figure to this path. Defaults to None.

    Returns:
        plt.Figure: The figure object containing the plot.
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

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the incidence curves
    ax.plot(days, exposed_incidence, "r-", label="Exposed")
    ax.plot(days, unexposed_incidence, "b-", label="Unexposed")

    # Add labels and title
    ax.set_xlabel(f"Days after index date + {offset_days} days")
    ax.set_ylabel("Cumulative Incidence")
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

    return fig


def plot_cumulative_incidence_simple(
    outcome_df: pd.DataFrame,
    index_date_df: pd.DataFrame,
    exposed_ids: list,
    offset_days: int = 0,
    max_follow_up_days: int = 365,
    at_risk_time_points: list = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
) -> plt.Figure:
    """
    Convenience function that combines data preparation and plotting.

    Parameters are the same as the individual functions.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot.
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
    )


# Example usage with sample data
if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)

    # Create sample data
    # Let's simulate 100 exposed subjects and 200 unexposed subjects
    n_exposed = 100
    n_unexposed = 200

    # Create index_date dataframe
    exposed_ids = list(range(1, n_exposed + 1))
    unexposed_ids = list(range(n_exposed + 1, n_exposed + n_unexposed + 1))
    all_ids = exposed_ids + unexposed_ids

    # Start dates randomly distributed over a year
    start_date = datetime(2023, 1, 1)
    index_dates = [
        start_date + timedelta(days=np.random.randint(0, 365))
        for _ in range(len(all_ids))
    ]

    index_date_df = pd.DataFrame({"subject_id": all_ids, "time": index_dates})

    # Create outcome dataframe
    # Not all subjects will have an outcome
    outcomes = []

    # Higher incidence and earlier outcomes for exposed group
    for subject_id in exposed_ids:
        if np.random.random() < 0.7:  # 70% chance of outcome for exposed
            index_time = index_date_df[index_date_df["subject_id"] == subject_id][
                "time"
            ].iloc[0]
            # Outcomes occur within 0-300 days after index date, with most in the earlier part
            outcome_time = index_time + timedelta(days=int(np.random.exponential(100)))
            outcomes.append({"subject_id": subject_id, "time": outcome_time})

    # Lower incidence and later outcomes for unexposed group
    for subject_id in unexposed_ids:
        if np.random.random() < 0.3:  # 30% chance of outcome for unexposed
            index_time = index_date_df[index_date_df["subject_id"] == subject_id][
                "time"
            ].iloc[0]
            # Outcomes occur within 0-300 days after index date, more evenly distributed
            outcome_time = index_time + timedelta(days=int(np.random.exponential(200)))
            outcomes.append({"subject_id": subject_id, "time": outcome_time})

    outcome_df = pd.DataFrame(outcomes)

    print("Example 1: Using the combined convenience function")
    # Plot with different offsets to demonstrate functionality using the combined function
    offsets_to_try = [0, 30, 90]
    for offset in offsets_to_try:
        fig = plot_cumulative_incidence_simple(
            outcome_df=outcome_df,
            index_date_df=index_date_df,
            exposed_ids=exposed_ids,
            offset_days=offset,
            save_path=f"figs/cumulative_incidence_offset_{offset}.png",
        )
        plt.close(fig)  # Close to avoid showing multiple plots at once

    print("\nExample 2: Using the separate functions for more control")
    # Example of using the separate functions for more control
    offset = 60

    # 1. Prepare the data
    incidence_data = prepare_incidence_data(
        outcome_df=outcome_df,
        index_date_df=index_date_df,
        exposed_ids=exposed_ids,
        offset_days=offset,
    )

    # Optional: You can inspect or modify the data here
    print(f"Data prepared for offset {offset} days")
    print(f"Maximum follow-up day: {max(incidence_data['days'])}")
    print(
        f"Final cumulative incidence in exposed group: {incidence_data['exposed_incidence'][-1]:.2%}"
    )
    print(
        f"Final cumulative incidence in unexposed group: {incidence_data['unexposed_incidence'][-1]:.2%}"
    )

    # 2. Create the plot with custom risk table timepoints
    fig = plot_cumulative_incidence(
        incidence_data=incidence_data,
        at_risk_time_points=[0, 30, 90, 180, 270],  # Custom timepoints
        figsize=(12, 7),  # Larger figure
        save_path=f"figs/cumulative_incidence_detailed_offset_{offset}.png",
    )

    # Show the last plot (uncomment to display interactively)
    # plt.show()

    print(f"\nGenerated cumulative incidence plots with multiple approaches")
