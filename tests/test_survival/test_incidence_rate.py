import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ehrviz.survival.incidence_rate import (
    plot_cumulative_incidence,
    plot_cumulative_incidence_simple,
)
from ehrviz.survival.prepare import prepare_incidence_data

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
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Plot with different offsets to demonstrate functionality
    offsets_to_try = [0, 30, 60, 90]
    for i, offset in enumerate(offsets_to_try):
        fig, axes[i] = plot_cumulative_incidence_simple(
            outcome_df=outcome_df,
            index_date_df=index_date_df,
            exposed_ids=exposed_ids,
            offset_days=offset,
            ax=axes[i],
        )
        axes[i].set_title(f"Offset: {offset} days")

    plt.tight_layout()

    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/cumulative_incidence_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    fig, ax = plot_cumulative_incidence(
        incidence_data=incidence_data,
        at_risk_time_points=[0, 30, 90, 180, 270],  # Custom timepoints
        figsize=(12, 7),  # Larger figure
        save_path=f"figs/cumulative_incidence_detailed_offset_{offset}.png",
    )

    # Show the last plot (uncomment to display interactively)
    # plt.show()

    print(f"\nGenerated cumulative incidence plots with multiple approaches")
