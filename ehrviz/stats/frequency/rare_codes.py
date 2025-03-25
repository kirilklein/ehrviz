from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ehrviz.stats.frequency.utils import filter_counts_by_pattern


def compute_rare_code_fraction(
    counts: pd.Series, n_range: Tuple[int, int] = (1, 100)
) -> Tuple[np.ndarray, list]:
    """
    Compute the fraction of codes with occurrences below each threshold.

    For each threshold n in the range from n_range[0] to n_range[1] (inclusive),
    calculates the fraction of unique codes whose count is less than n.

    Args:
        counts (pd.Series): A Series containing counts of codes
        n_range (tuple): A tuple specifying the (min, max) threshold values to evaluate.
                        Defaults to (1, 100).

    Returns:
        tuple:
            - x_values (np.ndarray): Array of threshold values
            - rare_fraction (list): List of fractions corresponding to each threshold

    Example:
        >>> counts = pd.Series({'A': 1, 'B': 2, 'C': 5, 'D': 10})
        >>> x_vals, fractions = compute_rare_code_fraction(counts, n_range=(1, 5))
        >>> print(f"Thresholds: {x_vals}")
        >>> print(f"Fractions: {fractions}")
        Thresholds: [1 2 3 4 5]
        Fractions: [0.0, 0.25, 0.5, 0.5, 0.75]
    """
    total_codes = len(counts)
    x_values = np.arange(n_range[0], n_range[1] + 1)
    rare_fraction = [np.sum(counts < n) / total_codes for n in x_values]
    return x_values, rare_fraction


def plot_rare_code_fractions(
    counts: pd.Series,
    groups: Dict[str, str],
    n_range: Tuple[int, int] = (1, 100),
    figsize: Tuple[int, int] = (12, 8),
    style: str = "seaborn-v0_8-darkgrid",
    title: str = "Rare Code Fraction vs. Threshold n",
    xlabel: str = "Threshold (n)",
    ylabel: str = "Fraction of Codes with < n Occurrences",
    legend_title: str = "Code Group",
    legend_ncols: int = 3,
    save_path: Union[str, None] = None,
    dpi: int = 300,
) -> None:
    """
    Plot the fraction of rare codes for various groups as a function of the occurrence threshold.

    This function filters the input counts for each group based on a regex pattern,
    computes the fraction of rare codes (codes with counts less than each threshold),
    and plots these fractions on a common set of axes for comparison.

    Args:
        counts (pd.Series): Series containing counts indexed by codes
        groups (dict): Dictionary mapping group labels to regex patterns for filtering codes
                      Example: {
                          "All": ".*",
                          "Lab": "^L/",
                          "Diagnoses": "^D/",
                          "Medication": "^M/"
                      }
        n_range (tuple): Tuple specifying the (min, max) thresholds to evaluate.
                        Defaults to (1, 100).
        figsize (tuple): Figure dimensions (width, height) in inches.
                        Defaults to (12, 8).
        style (str): Matplotlib style to use. Defaults to 'seaborn-v0_8-darkgrid'.
        title (str): Plot title. Defaults to "Rare Code Fraction vs. Threshold n".
        xlabel (str): X-axis label. Defaults to "Threshold (n)".
        ylabel (str): Y-axis label. Defaults to "Fraction of Codes with < n Occurrences".
        legend_title (str): Title for the legend. Defaults to "Code Group".
        legend_ncols (int): Number of columns in the legend. Defaults to 3.
        save_path (str, optional): Path to save the figure. If None, figure is not saved.
        dpi (int): DPI for saved figure. Defaults to 300.

    Returns:
        None: Displays and optionally saves a matplotlib plot
    """
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=figsize)

    # Process and plot each group
    for group_label, regex_pattern in groups.items():
        # Filter the counts using the regex pattern
        group_counts = filter_counts_by_pattern(counts, regex_pattern)

        if group_counts.empty:
            continue  # Skip empty groups

        # Compute the rare code fraction for thresholds in the given range
        x_vals, rare_fraction = compute_rare_code_fraction(
            group_counts, n_range=n_range
        )

        # Plot the rare code fraction for this group
        ax.plot(
            x_vals,
            rare_fraction,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=group_label,
        )

    # Configure plot appearance
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(title=legend_title, fontsize=10, ncol=legend_ncols)
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    plt.show()
