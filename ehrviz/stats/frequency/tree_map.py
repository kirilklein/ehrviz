"""
Module for creating treemap visualizations of code frequencies.

This module provides functionality to create treemap visualizations showing the relative
frequencies of medical codes in a dataset. It includes utilities for preparing the data
and generating the visualization.

The treemap visualization represents each code as a rectangle, where the area of the
rectangle is proportional to the frequency of that code in the dataset. This provides
an intuitive way to visualize the distribution of codes.

Example:
    >>> counts = {'CODE1': 0.5, 'CODE2': 0.3, 'CODE3': 0.2}
    >>> sizes, codes = prepare_treemap_data(counts)
    >>> plot_treemap(sizes, codes)

Functions:
    prepare_treemap_data: Prepares count data for treemap visualization
    plot_treemap: Creates and displays the treemap visualization
"""

import matplotlib.pyplot as plt
from ehrviz.stats.frequency.utils import (
    prepare_treemap_data,
    compute_treemap_rectangles,
)


def draw_treemap(ax, rects, codes, sizes, big_threshold=6, medium_threshold=5):
    """
    Draw treemap rectangles with annotated labels on a given Axes.

    The function iterates through each rectangle and adds it to the plot with a
    distinct color. Depending on the rectangle dimensions, it annotates each
    rectangle with either both the code and its percentage (for large rectangles)
    or just the code (for medium rectangles).

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object on which to draw the treemap.
        rects (list): List of rectangle dictionaries with keys 'x', 'y', 'dx', 'dy'.
        codes (list): List of labels corresponding to each rectangle.
        sizes (np.ndarray): Array of percentage values corresponding to each rectangle.
        big_threshold (float, optional): Minimum rectangle dimension for displaying
            both code and percentage. Defaults to 6.
        medium_threshold (float, optional): Minimum rectangle dimension for displaying
            only the code. Defaults to 5.
    """
    # Generate a discrete color for each rectangle using the tab20 colormap.
    colors = [plt.cm.tab20(i % 20) for i in range(len(rects))]

    for rect, code, pct, color in zip(rects, codes, sizes, colors):
        # Draw the rectangle with white borders.
        ax.add_patch(
            plt.Rectangle(
                (rect["x"], rect["y"]),
                rect["dx"],
                rect["dy"],
                facecolor=color,
                edgecolor="white",
                linewidth=2,
            )
        )

        # Determine the label based on rectangle size.
        if rect["dx"] >= big_threshold and rect["dy"] >= big_threshold:
            label_text = f"{code}\n{pct:.1f}%"
        elif rect["dx"] >= medium_threshold and rect["dy"] >= medium_threshold:
            label_text = f"{code}"
        else:
            label_text = ""

        # Add the label if there's text to display.
        if label_text:
            ax.text(
                rect["x"] + rect["dx"] / 2,
                rect["y"] + rect["dy"] / 2,
                label_text,
                ha="center",
                va="center",
                fontsize=10,
            )


def plot_treemap(
    counts,
    rename_dict=None,
    big_threshold=6,
    medium_threshold=5,
    save_path=None,
    dpi=300,
    width=100,
    height=100,
    figsize=(12, 8),
    axis_off=True,
    title="Treemap of Code Types (Percentage)",
    title_fontsize=16,
):
    """
    Plot a treemap visualization of code counts with annotated labels.

    The function processes the provided counts by converting them to percentages,
    computes the treemap layout using squarify, and draws the treemap using matplotlib.
    Depending on the size of each rectangle:
      - Large rectangles (meeting big_threshold) display both the code label and the percentage.
      - Medium rectangles (meeting medium_threshold) display only the code label.
      - Small rectangles are left unannotated.

    Parameters:
        counts (pd.Series): A pandas Series with index as labels and values as counts.
        rename_dict (dict, optional): A dictionary to rename the labels for display purposes.
            Defaults to None.
        big_threshold (float, optional): Minimum rectangle dimension for full annotation
            (label and percentage). Defaults to 6.
        medium_threshold (float, optional): Minimum rectangle dimension for label-only annotation.
            Defaults to 5.

    Returns:
        None. The function displays the treemap plot.
    """
    # Prepare the data: convert counts to percentages and extract labels.
    sizes, codes = prepare_treemap_data(counts, rename_dict)

    # Compute the layout of the treemap rectangles.
    rects = compute_treemap_rectangles(sizes, width, height)

    # Set up the plot.
    fig, ax = plt.subplots(figsize=figsize)

    # Draw the treemap on the Axes.
    draw_treemap(ax, rects, codes, sizes, big_threshold, medium_threshold)

    # Configure the plot appearance.
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    if axis_off:
        ax.axis("off")
    ax.set_title(title, fontsize=title_fontsize)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.show()
