from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def plot_cumulative_incidence(
    cumulative_incidence: pd.DataFrame,
    label: str,
    fig=None,
    ax=None,
    color=None,
    linestyle="-",
    linewidth=2,
    xlabel="Days Since Index Date (+offset)",
    ylabel="Cumulative Incidence Rate (%)",
    title="Cumulative Incidence Analysis",
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot cumulative incidence curve.

    Parameters:
    -----------
    cumulative_incidence : pd.DataFrame
        Output from calculate_cumulative_incidence
    label : str
        Legend label for this curve
    fig, ax : matplotlib objects, optional
        Existing figure and axes to plot on
    color : str, optional
        Line color
    linestyle : str, default='-'
        Line style
    linewidth : float, default=2
        Line width
    **kwargs : dict
        Additional arguments passed to plt.plot()

    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the cumulative incidence
    ax.plot(
        cumulative_incidence["day"],
        cumulative_incidence["cumulative_incidence"],
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        **kwargs,
    )

    # Formatting
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Add some styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig, ax
