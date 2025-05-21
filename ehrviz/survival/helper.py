import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def compute_risk_table(
    valid_outcomes: pd.DataFrame,
    total_exposed: int,
    total_unexposed: int,
    time_points: list,
    max_days: int,
) -> dict:
    """
    Compute the risk table data for specified time points.

    Args:
        valid_outcomes (pd.DataFrame): DataFrame containing valid outcomes
        total_exposed (int): Total number of exposed subjects
        total_unexposed (int): Total number of unexposed subjects
        time_points (list): Time points to compute risk table for
        max_days (int): Maximum number of days in the study

    Returns:
        dict: Dictionary containing risk table data with keys:
            - time_points: List of filtered time points
            - exposed_at_risk: List of exposed subjects at risk at each time point
            - unexposed_at_risk: List of unexposed subjects at risk at each time point
    """
    # Filter time points
    filtered_time_points = [tp for tp in time_points if tp <= max_days]

    exposed_at_risk = []
    unexposed_at_risk = []

    for day in filtered_time_points:
        # Count events by this day
        exposed_events = len(
            valid_outcomes[
                (valid_outcomes["exposure_status"] == "Exposed")
                & (valid_outcomes["days_to_outcome"] <= day)
            ]["subject_id"].unique()
        )
        unexposed_events = len(
            valid_outcomes[
                (valid_outcomes["exposure_status"] == "Unexposed")
                & (valid_outcomes["days_to_outcome"] <= day)
            ]["subject_id"].unique()
        )

        # Calculate subjects at risk
        exposed_at_risk.append(total_exposed - exposed_events)
        unexposed_at_risk.append(total_unexposed - unexposed_events)

    return {
        "time_points": filtered_time_points,
        "exposed_at_risk": exposed_at_risk,
        "unexposed_at_risk": unexposed_at_risk,
    }


def add_risk_table_to_plot(
    ax: plt.Axes, risk_table_data: dict, fig: plt.Figure
) -> None:
    """
    Add risk table to the plot.

    Args:
        ax (plt.Axes): The axes object to add the table to
        risk_table_data (dict): Dictionary containing risk table data
        fig (plt.Figure): The figure object
    """
    # Create space for table
    fig.subplots_adjust(bottom=0.25)

    # Calculate positions
    n_points = len(risk_table_data["time_points"])
    # Start table after labels (0.15) and use remaining space (0.85)
    x_positions = np.linspace(0.15, 0.85, n_points)

    # Add headers with fixed position
    ax.text(0.02, -0.15, "Days:", transform=ax.transAxes, ha="left")
    ax.text(0.02, -0.20, "Exposed:", transform=ax.transAxes, ha="left")
    ax.text(0.02, -0.25, "Unexposed:", transform=ax.transAxes, ha="left")

    # Fill table
    for i, (day, exp_risk, unexp_risk) in enumerate(
        zip(
            risk_table_data["time_points"],
            risk_table_data["exposed_at_risk"],
            risk_table_data["unexposed_at_risk"],
        )
    ):
        # Add to table with proper spacing
        ax.text(x_positions[i], -0.15, str(day), transform=ax.transAxes, ha="center")
        ax.text(
            x_positions[i], -0.20, str(exp_risk), transform=ax.transAxes, ha="center"
        )
        ax.text(
            x_positions[i], -0.25, str(unexp_risk), transform=ax.transAxes, ha="center"
        )
