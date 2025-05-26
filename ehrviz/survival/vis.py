import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from lifelines import KaplanMeierFitter
from typing import Optional, Dict, Any


def plot_kaplan_meier_cumulative_density(
    lifelines_data: pd.DataFrame,
    ax: plt.Axes,
    label: str,
    weights: Optional[pd.Series] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    plot_kwargs: Optional[Dict[str, Any]] = None,
) -> plt.Axes:
    """
    Plot Kaplan-Meier cumulative incidence curve using lifelines.

    This function creates a cumulative incidence plot (1 - survival) using the
    Kaplan-Meier estimator. The plot shows the probability of experiencing the
    event of interest over time.

    Parameters
    ----------
    lifelines_data : pd.DataFrame
        Survival data with columns:
        - 'T': Duration until event or censoring (numeric)
        - 'E': Event indicator (1 = event occurred, 0 = censored)
        - 'subject_id': Subject identifier (optional, for reference)

    ax : plt.Axes
        Matplotlib axes object to plot on

    label : str
        Label for the curve in the legend

    weights : pd.Series, optional
        Weights for each observation, must be same length as lifelines_data.
        Used for weighted Kaplan-Meier estimation (e.g., inverse probability weights).
        If None, all observations are weighted equally.

    fit_kwargs : dict, optional
        Additional keyword arguments passed to KaplanMeierFitter.fit().
        Common options:
        - alpha: Confidence level for confidence intervals (default 0.05)
        - ci_labels: Custom labels for confidence intervals

    plot_kwargs : dict, optional
        Additional keyword arguments passed to plot_cumulative_density().
        Common options:
        - at_risk_counts: Show number at risk table (bool, default False)
        - show_censors: Show censoring marks (bool, default True)
        - ci_show: Show confidence intervals (bool, default True)
        - color: Line color
        - linestyle: Line style

    Returns
    -------
    plt.Axes
        The modified axes object with the plot

    Examples
    --------
    Basic usage:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> plot_kaplan_meier_cumulative_density(
    ...     lifelines_data, ax, "Treatment Group"
    ... )
    >>> plt.show()

    With weights and custom styling:

    >>> weights = lifelines_data['subject_id'].map(weight_dict)
    >>> plot_kaplan_meier_cumulative_density(
    ...     lifelines_data, ax, "Weighted Analysis",
    ...     weights=weights,
    ...     plot_kwargs={'at_risk_counts': True, 'color': 'red'}
    ... )

    Comparing multiple groups:

    >>> fig, ax = plt.subplots(figsize=(10, 6))
    >>> for group_name, group_data in data_by_group.items():
    ...     plot_kaplan_meier_cumulative_density(
    ...         group_data, ax, f"Group {group_name}"
    ...     )
    >>> ax.legend()
    >>> plt.show()

    Notes
    -----
    - The function plots cumulative incidence (1 - survival probability)
    - Y-axis is automatically formatted as percentages (0-100%)
    - Confidence intervals are shown by default if available
    - The plot shows the probability of experiencing the event by time t

    See Also
    --------
    prepare_survival_data_for_lifelines : Prepare data in the required format
    lifelines.KaplanMeierFitter : Underlying estimation method
    """
    # Initialize default arguments
    if fit_kwargs is None:
        fit_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    # Add weights to fit_kwargs if provided
    if weights is not None:
        fit_kwargs["weights"] = weights

    # Create and fit Kaplan-Meier estimator
    kmf = KaplanMeierFitter(label=label)
    kmf.fit(lifelines_data["T"], lifelines_data["E"], **fit_kwargs)

    # Plot cumulative density (cumulative incidence)
    kmf.plot_cumulative_density(ax=ax, **plot_kwargs)

    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Cumulative Incidence (%)")

    return ax
