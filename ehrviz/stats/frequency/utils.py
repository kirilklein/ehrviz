from typing import Union

import pandas as pd
import squarify


def filter_counts_by_pattern(
    counts: pd.Series, pattern: str, case: bool = True
) -> pd.Series:
    """
    Filter counts Series based on a regex pattern.

    Args:
        counts (pd.Series): Series containing counts indexed by codes
        pattern (str): Regex pattern to filter codes
        case (bool, optional): Whether to consider case in regex matching. Defaults to True.

    Returns:
        pd.Series: Filtered counts Series containing only matching codes
    """
    matching_codes = counts.index.str.contains(pattern, regex=True, case=case)
    return counts[matching_codes]


def group_counts(counts: pd.Series, rename_dict: dict) -> pd.Series:
    """
    Groups and aggregates count data based on a mapping dictionary.

    This function takes a Series of counts indexed by codes and consolidates them
    according to a provided mapping dictionary. Codes that share the same mapped name
    will have their counts summed together.

    Args:
        counts (pandas.Series): A Series containing count data, indexed by codes
        rename_dict (dict): A dictionary mapping original codes to new names
                           {original_code: new_name}

    Returns:
        pandas.Series: A new Series with grouped and summed counts, sorted in
                      descending order by value

    Example:
        >>> counts = pd.Series({
        ...     'A1': 10,
        ...     'A2': 15,
        ...     'B1': 5
        ... })
        >>> rename_dict = {'A1': 'A', 'A2': 'A', 'B1': 'B'}
        >>> group_counts(counts, rename_dict)
        A    25
        B     5
        dtype: int64
    """
    # Create new index using the mapping dictionary
    new_index = [rename_dict.get(code, code) for code in counts.index]

    # Create a new Series with the mapped index and group by new names
    grouped_counts = counts.copy()
    grouped_counts.index = new_index
    return grouped_counts.groupby(level=0).sum().sort_values(ascending=False)


def prepare_treemap_data(counts: pd.Series, rename_dict: Union[dict, None] = None):
    """
    Prepare data for treemap visualization.

    This function converts raw counts to percentages, extracts the size values
    and the corresponding code labels, and optionally renames the codes using
    the provided mapping. It supports both dictionaries and pandas Series as input.

    Parameters:
        counts (dict or pd.Series): Input counts where the keys/index represent codes
            and the values represent counts.
        rename_dict (dict, optional): A dictionary to map original code labels
            to alternative display names. Defaults to None.

    Returns:
        tuple:
            sizes (np.ndarray): Array of percentage values computed from counts.
            codes (list): List of code labels (renamed if rename_dict is provided).
    """
    # Convert dictionary to Series if needed
    if isinstance(counts, dict):
        counts = pd.Series(counts)
    # Assume counts is a pandas Series
    percentages = counts * 100
    sizes = percentages.values
    codes = list(percentages.index)

    if rename_dict is not None:
        codes = [rename_dict.get(code, code) for code in codes]

    return sizes, codes


def compute_treemap_rectangles(sizes, width=100, height=100):
    """
    Compute normalized treemap rectangles for given sizes within a defined area.

    The function uses squarify to normalize the sizes so that the sum of the areas
    fits within a rectangle of size width x height, and then computes the layout.

    Parameters:
        sizes (np.ndarray): Array of size values (as percentages).
        width (float, optional): The width of the treemap area. Defaults to 100.
        height (float, optional): The height of the treemap area. Defaults to 100.

    Returns:
        list: A list of dictionaries. Each dictionary has keys 'x', 'y', 'dx', 'dy'
              corresponding to the position and size of a rectangle.
    """
    norm_sizes = squarify.normalize_sizes(sizes, width, height)
    rects = squarify.squarify(norm_sizes, 0, 0, width, height)
    return rects
