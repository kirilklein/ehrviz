import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ehrviz.survival.incidence_rate import plot_cumulative_incidence
from ehrviz.survival.prepare import calculate_cumulative_incidence_groups

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

    # Create a directory for saving figures
    os.makedirs("figs", exist_ok=True)

    print("Example 1: Using the new modular approach with multiple offsets")
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Plot with different offsets to demonstrate functionality
    offsets_to_try = [0, 30, 60, 90]

    # Define groups
    groups = {"Exposed": exposed_ids, "Unexposed": unexposed_ids}

    # Define colors for consistency
    group_colors = {"Exposed": "red", "Unexposed": "blue"}

    for i, offset in enumerate(offsets_to_try):
        # Calculate incidence for both groups with the current offset
        incidence_results = calculate_cumulative_incidence_groups(
            group_definitions=groups,
            outcome_df=outcome_df,
            index_date_df=index_date_df,
            offset_days=offset,
            max_follow_up_days=365,
        )

        # Plot on the current subplot
        plot_cumulative_incidence(
            incidence_data=incidence_results,
            group_colors=group_colors,
            offset_days=offset,
            title=f"Offset: {offset} days",
            ax=axes[i],
        )

    plt.tight_layout()
    plt.savefig("figs/cumulative_incidence_grid_new.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nExample 2: Detailed examination of a single offset")
    # Example with a single offset for more detailed examination
    offset = 60

    # Calculate incidence with this offset
    incidence_results = calculate_cumulative_incidence_groups(
        group_definitions=groups,
        outcome_df=outcome_df,
        index_date_df=index_date_df,
        offset_days=offset,
        max_follow_up_days=365,
    )

    # Print some information about the results
    print(f"Data prepared for offset {offset} days")
    for group_name, df in incidence_results.items():
        max_day = df["day"].max()
        final_incidence = df["cumulative_incidence"].iloc[-1]
        print(f"Group '{group_name}':")
        print(f"  - Maximum follow-up day: {max_day}")
        print(f"  - Final cumulative incidence: {final_incidence:.2%}")

    # Create a detailed plot
    fig, ax = plot_cumulative_incidence(
        incidence_data=incidence_results,
        group_colors=group_colors,
        offset_days=offset,
        figsize=(12, 7),
        save_path=f"figs/cumulative_incidence_detailed_offset_{offset}_new.png",
    )

    print("\nExample 3: Using individual follow-up periods")
    # Create follow-up dates for some subjects
    # Simulate that some subjects drop out early
    follow_up_data = []

    for subject_id in all_ids:
        # 20% chance of early follow-up end
        if np.random.random() < 0.2:
            index_time = index_date_df[index_date_df["subject_id"] == subject_id][
                "time"
            ].iloc[0]
            # Follow-up ends between 30-180 days
            follow_up_end = index_time + timedelta(days=np.random.randint(30, 180))
            follow_up_data.append({"subject_id": subject_id, "end_date": follow_up_end})

    follow_up_df = pd.DataFrame(follow_up_data) if follow_up_data else None

    # Calculate incidence with individual follow-up
    incidence_with_followup = calculate_cumulative_incidence_groups(
        group_definitions=groups,
        outcome_df=outcome_df,
        index_date_df=index_date_df,
        offset_days=offset,
        max_follow_up_days=365,
        follow_up_df=follow_up_df,
    )

    # Plot both the standard and follow-up adjusted incidence rates
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the standard incidence (dashed lines)
    for group_name, df in incidence_results.items():
        ax.plot(
            df["day"],
            df["cumulative_incidence"],
            color=group_colors.get(group_name, "black"),
            linestyle="--",
            label=f"{group_name} (Standard)",
        )

    # Plot the follow-up adjusted incidence (solid lines)
    for group_name, df in incidence_with_followup.items():
        ax.plot(
            df["day"],
            df["cumulative_incidence"],
            color=group_colors.get(group_name, "black"),
            linestyle="-",
            label=f"{group_name} (With individual follow-up)",
        )

    ax.set_title(
        f"Comparison of Standard vs. Individual Follow-up (Offset: {offset} days)"
    )
    ax.set_xlabel(f"Days after index date + {offset} days")
    ax.set_ylabel("Cumulative Incidence Rate (%)")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        "figs/cumulative_incidence_with_followup.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    print("\nExample 4: Using weights")
    # Create sample weights
    weights = {}
    for subject_id in all_ids:
        # Assign random weights between 0.5 and 1.5
        weights[subject_id] = 0.5 + np.random.random()

    # Calculate weighted incidence
    weighted_incidence = calculate_cumulative_incidence_groups(
        group_definitions=groups,
        outcome_df=outcome_df,
        index_date_df=index_date_df,
        offset_days=offset,
        max_follow_up_days=365,
        weights=weights,
    )

    print("\nExample 5: Comparing three groups")
    # Let's create a third 'high-risk' group from the exposed group
    high_risk_ids = exposed_ids[:30]  # First 30 of the exposed group
    medium_risk_ids = exposed_ids[30:]  # Remaining exposed group
    low_risk_ids = unexposed_ids  # Unexposed group

    # Define the three groups
    three_groups = {
        "High Risk": high_risk_ids,
        "Medium Risk": medium_risk_ids,
        "Low Risk": low_risk_ids,
    }

    # Define colors for the three groups
    three_group_colors = {
        "High Risk": "red",
        "Medium Risk": "orange",
        "Low Risk": "blue",
    }

    # Calculate incidence for three groups
    three_group_results = calculate_cumulative_incidence_groups(
        group_definitions=three_groups,
        outcome_df=outcome_df,
        index_date_df=index_date_df,
        offset_days=offset,
        max_follow_up_days=365,
    )

    # Plot the three-group comparison
    fig, ax = plot_cumulative_incidence(
        incidence_data=three_group_results,
        group_colors=three_group_colors,
        offset_days=offset,
        title="Comparison of Three Risk Groups",
        figsize=(12, 7),
        save_path="figs/three_group_comparison.png",
    )

    print(
        f"\nGenerated multiple cumulative incidence plots with the new modular approach"
    )
