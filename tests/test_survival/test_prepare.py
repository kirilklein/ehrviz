import unittest

import pandas as pd

from ehrviz.survival.prepare import (
    _create_daily_records,
    _determine_events,
    _ensure_datetime_columns,
    _find_first_outcomes,
    _prepare_subjects_baseline,
    prepare_survival_data,
)


class TestEnsureDatetimeColumns(unittest.TestCase):
    """Test datetime column conversion."""

    def test_converts_string_dates(self):
        df = pd.DataFrame(
            {"subject_id": ["A", "B"], "time": ["2020-01-01", "2020-01-02"]}
        )
        _ensure_datetime_columns([df])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["time"]))

    def test_handles_missing_time_column(self):
        df = pd.DataFrame({"subject_id": ["A", "B"]})
        # Should not raise error
        _ensure_datetime_columns([df])
        self.assertNotIn("time", df.columns)

    def test_handles_already_datetime(self):
        df = pd.DataFrame(
            {
                "subject_id": ["A", "B"],
                "time": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            }
        )
        original_dtype = df["time"].dtype
        _ensure_datetime_columns([df])
        self.assertEqual(df["time"].dtype, original_dtype)


class TestPrepareSubjectsBaseline(unittest.TestCase):
    """Test baseline subject data preparation."""

    def test_basic_merge_and_calculations(self):
        index_dates = pd.DataFrame(
            {
                "subject_id": ["A", "B"],
                "time": pd.to_datetime(["2020-01-01", "2020-01-05"]),
            }
        )
        censor_dates = pd.DataFrame(
            {
                "subject_id": ["A", "B"],
                "time": pd.to_datetime(["2020-01-10", "2020-01-15"]),
            }
        )

        result = _prepare_subjects_baseline(index_dates, censor_dates, offset_days=0)

        self.assertEqual(len(result), 2)
        self.assertIn("observation_start", result.columns)
        self.assertIn("censor_day", result.columns)

        # Check calculations
        self.assertEqual(result.loc[0, "censor_day"], 9)  # 10 days difference
        self.assertEqual(result.loc[1, "censor_day"], 10)  # 10 days difference

    def test_offset_days(self):
        index_dates = pd.DataFrame(
            {"subject_id": ["A"], "time": pd.to_datetime(["2020-01-01"])}
        )
        censor_dates = pd.DataFrame(
            {"subject_id": ["A"], "time": pd.to_datetime(["2020-01-10"])}
        )

        result = _prepare_subjects_baseline(index_dates, censor_dates, offset_days=2)

        expected_obs_start = pd.to_datetime("2020-01-03")
        self.assertEqual(result.loc[0, "observation_start"], expected_obs_start)
        self.assertEqual(result.loc[0, "censor_day"], 7)  # 10 - 3 = 7 days

    def test_negative_censor_day_clipped(self):
        index_dates = pd.DataFrame(
            {"subject_id": ["A"], "time": pd.to_datetime(["2020-01-10"])}
        )
        censor_dates = pd.DataFrame(
            {
                "subject_id": ["A"],
                "time": pd.to_datetime(["2020-01-05"]),  # Before index date
            }
        )

        result = _prepare_subjects_baseline(index_dates, censor_dates, offset_days=0)

        self.assertEqual(result.loc[0, "censor_day"], 0)  # Clipped to 0


class TestFindFirstOutcomes(unittest.TestCase):
    """Test outcome finding logic."""

    def test_no_outcomes(self):
        outcomes = pd.DataFrame(columns=["subject_id", "time"])
        subjects_df = pd.DataFrame(
            {
                "subject_id": ["A", "B"],
                "observation_start": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            }
        )

        result = _find_first_outcomes(outcomes, subjects_df)

        self.assertTrue(result["first_outcome_time"].isna().all())

    def test_outcomes_after_observation_start(self):
        outcomes = pd.DataFrame(
            {
                "subject_id": ["A", "A", "B"],
                "time": pd.to_datetime(["2020-01-05", "2020-01-10", "2020-01-08"]),
            }
        )
        subjects_df = pd.DataFrame(
            {
                "subject_id": ["A", "B"],
                "observation_start": pd.to_datetime(["2020-01-03", "2020-01-03"]),
            }
        )

        result = _find_first_outcomes(outcomes, subjects_df)

        # Subject A should have first outcome on 2020-01-05
        self.assertEqual(
            result.loc[0, "first_outcome_time"], pd.to_datetime("2020-01-05")
        )
        # Subject B should have outcome on 2020-01-08
        self.assertEqual(
            result.loc[1, "first_outcome_time"], pd.to_datetime("2020-01-08")
        )

    def test_outcomes_before_observation_start_ignored(self):
        outcomes = pd.DataFrame(
            {
                "subject_id": ["A"],
                "time": pd.to_datetime(["2020-01-01"]),  # Before observation start
            }
        )
        subjects_df = pd.DataFrame(
            {"subject_id": ["A"], "observation_start": pd.to_datetime(["2020-01-05"])}
        )

        result = _find_first_outcomes(outcomes, subjects_df)

        self.assertTrue(result.loc[0, "first_outcome_time"] is pd.NaT)


class TestDetermineEvents(unittest.TestCase):
    """Test event determination logic."""

    def test_outcome_before_censor(self):
        subjects_df = pd.DataFrame(
            {
                "subject_id": ["A"],
                "observation_start": pd.to_datetime(["2020-01-01"]),
                "censor_day": [10],
                "time_censor": pd.to_datetime(["2020-01-11"]),
                "first_outcome_time": pd.to_datetime(["2020-01-05"]),
            }
        )

        result = _determine_events(subjects_df, max_follow_up_days=None)

        self.assertEqual(result.loc[0, "event_day"], 4)  # Outcome on day 4
        self.assertTrue(result.loc[0, "is_outcome"])

    def test_censor_before_outcome(self):
        subjects_df = pd.DataFrame(
            {
                "subject_id": ["A"],
                "observation_start": pd.to_datetime(["2020-01-01"]),
                "censor_day": [5],
                "time_censor": pd.to_datetime(["2020-01-06"]),
                "first_outcome_time": pd.to_datetime(["2020-01-10"]),
            }
        )

        result = _determine_events(subjects_df, max_follow_up_days=None)

        self.assertEqual(result.loc[0, "event_day"], 5)  # Censored on day 5
        self.assertFalse(result.loc[0, "is_outcome"])

    def test_no_outcome_only_censor(self):
        subjects_df = pd.DataFrame(
            {
                "subject_id": ["A"],
                "observation_start": pd.to_datetime(["2020-01-01"]),
                "censor_day": [10],
                "time_censor": pd.to_datetime(["2020-01-11"]),
                "first_outcome_time": [pd.NaT],
            }
        )

        result = _determine_events(subjects_df, max_follow_up_days=None)

        self.assertEqual(result.loc[0, "event_day"], 10)
        self.assertFalse(result.loc[0, "is_outcome"])

    def test_max_follow_up_constraint(self):
        subjects_df = pd.DataFrame(
            {
                "subject_id": ["A"],
                "observation_start": pd.to_datetime(["2020-01-01"]),
                "censor_day": [100],
                "time_censor": pd.to_datetime(["2020-04-11"]),
                "first_outcome_time": [pd.NaT],
            }
        )

        result = _determine_events(subjects_df, max_follow_up_days=30)

        self.assertEqual(result.loc[0, "event_day"], 30)  # Clipped to max follow-up

    def test_negative_event_day_filtered(self):
        subjects_df = pd.DataFrame(
            {
                "subject_id": ["A", "B"],
                "observation_start": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "censor_day": [-5, 10],  # Negative day should be filtered
                "time_censor": pd.to_datetime(["2019-12-27", "2020-01-11"]),
                "first_outcome_time": [pd.NaT, pd.NaT],
            }
        )

        result = _determine_events(subjects_df, max_follow_up_days=None)

        self.assertEqual(len(result), 1)  # Only subject B should remain
        self.assertEqual(result.iloc[0]["subject_id"], "B")


class TestCreateDailyRecords(unittest.TestCase):
    """Test daily record creation."""

    def test_empty_input(self):
        subjects_df = pd.DataFrame(columns=["subject_id", "event_day", "is_outcome"])

        result = _create_daily_records(subjects_df)

        expected_columns = ["subject_id", "day", "censored", "outcome"]
        self.assertListEqual(list(result.columns), expected_columns)
        self.assertEqual(len(result), 0)

    def test_outcome_event(self):
        subjects_df = pd.DataFrame(
            {"subject_id": ["A"], "event_day": [3], "is_outcome": [True]}
        )

        result = _create_daily_records(subjects_df)

        self.assertEqual(len(result), 4)  # Days 0, 1, 2, 3
        self.assertEqual(result["outcome"].sum(), 1)  # Only day 3 has outcome
        self.assertEqual(result["censored"].sum(), 0)  # No censoring
        self.assertEqual(result.loc[3, "outcome"], 1)  # Day 3 has outcome

    def test_censoring_event(self):
        subjects_df = pd.DataFrame(
            {"subject_id": ["A"], "event_day": [2], "is_outcome": [False]}
        )

        result = _create_daily_records(subjects_df)

        self.assertEqual(len(result), 3)  # Days 0, 1, 2
        self.assertEqual(result["outcome"].sum(), 0)  # No outcomes
        self.assertEqual(result["censored"].sum(), 1)  # Only day 2 has censoring
        self.assertEqual(result.loc[2, "censored"], 1)  # Day 2 has censoring

    def test_multiple_subjects(self):
        subjects_df = pd.DataFrame(
            {"subject_id": ["A", "B"], "event_day": [1, 2], "is_outcome": [True, False]}
        )

        result = _create_daily_records(subjects_df)

        # Subject A: days 0, 1 (outcome on day 1)
        # Subject B: days 0, 1, 2 (censored on day 2)
        self.assertEqual(len(result), 5)

        subject_a_records = result[result["subject_id"] == "A"]
        subject_b_records = result[result["subject_id"] == "B"]

        self.assertEqual(len(subject_a_records), 2)
        self.assertEqual(len(subject_b_records), 3)
        self.assertEqual(subject_a_records["outcome"].sum(), 1)
        self.assertEqual(subject_b_records["censored"].sum(), 1)


class TestPrepareSurvivalDataIntegration(unittest.TestCase):
    """Integration tests for the main function."""

    def setUp(self):
        """Set up test data."""
        self.index_dates = pd.DataFrame(
            {
                "subject_id": ["A", "B", "C"],
                "time": ["2020-01-01", "2020-01-01", "2020-01-01"],
            }
        )

        self.censor_dates = pd.DataFrame(
            {
                "subject_id": ["A", "B", "C"],
                "time": ["2020-01-10", "2020-01-15", "2020-01-20"],
            }
        )

        self.outcomes = pd.DataFrame(
            {
                "subject_id": ["A", "C"],
                "time": ["2020-01-05", "2020-01-25"],  # C's outcome after censor
            }
        )

    def test_basic_functionality(self):
        """Test basic survival data preparation."""
        result = prepare_survival_data(
            self.index_dates, self.outcomes, self.censor_dates, offset_days=0
        )

        # Check structure
        expected_columns = ["subject_id", "day", "censored", "outcome"]
        self.assertListEqual(list(result.columns), expected_columns)

        # Check that all subjects are present
        subjects_in_result = result["subject_id"].unique()
        self.assertIn("A", subjects_in_result)
        self.assertIn("B", subjects_in_result)
        self.assertIn("C", subjects_in_result)

        # Subject A should have outcome on day 4
        subject_a = result[result["subject_id"] == "A"]
        self.assertEqual(subject_a["outcome"].sum(), 1)
        self.assertEqual(subject_a[subject_a["outcome"] == 1]["day"].iloc[0], 4)

        # Subject B should be censored (no outcome)
        subject_b = result[result["subject_id"] == "B"]
        self.assertEqual(subject_b["outcome"].sum(), 0)
        self.assertEqual(subject_b["censored"].sum(), 1)

        # Subject C should be censored (outcome after censor date)
        subject_c = result[result["subject_id"] == "C"]
        self.assertEqual(subject_c["outcome"].sum(), 0)
        self.assertEqual(subject_c["censored"].sum(), 1)

    def test_offset_days(self):
        """Test with offset days."""
        result = prepare_survival_data(
            self.index_dates, self.outcomes, self.censor_dates, offset_days=2
        )

        # Subject A outcome should now be on day 2 (was day 4, now 4-2=2)
        subject_a = result[result["subject_id"] == "A"]
        outcome_day = subject_a[subject_a["outcome"] == 1]["day"].iloc[0]
        self.assertEqual(outcome_day, 2)

    def test_max_follow_up(self):
        """Test with maximum follow-up constraint."""
        result = prepare_survival_data(
            self.index_dates,
            self.outcomes,
            self.censor_dates,
            offset_days=0,
            max_follow_up_days=5,
        )

        # No subject should have records beyond day 5
        max_day = result["day"].max()
        self.assertLessEqual(max_day, 5)

        # Subject B should be censored at day 5 instead of day 14
        subject_b = result[result["subject_id"] == "B"]
        max_day_b = subject_b["day"].max()
        self.assertEqual(max_day_b, 5)

    def test_empty_outcomes(self):
        """Test with no outcomes."""
        empty_outcomes = pd.DataFrame(columns=["subject_id", "time"])

        result = prepare_survival_data(
            self.index_dates, empty_outcomes, self.censor_dates, offset_days=0
        )

        # All subjects should be censored, no outcomes
        self.assertEqual(result["outcome"].sum(), 0)
        self.assertGreater(result["censored"].sum(), 0)

    def test_string_dates_converted(self):
        """Test that string dates are properly converted."""
        # Input data has string dates
        result = prepare_survival_data(
            self.index_dates, self.outcomes, self.censor_dates, offset_days=0
        )

        # Should work without errors and produce valid results
        self.assertGreater(len(result), 0)
        self.assertIn("subject_id", result.columns)

    def test_missing_subjects_in_censor_dates(self):
        """Test behavior when subjects missing from censor dates."""
        incomplete_censor = pd.DataFrame(
            {
                "subject_id": ["A", "B"],  # Missing subject C
                "time": ["2020-01-10", "2020-01-15"],
            }
        )

        result = prepare_survival_data(
            self.index_dates, self.outcomes, incomplete_censor, offset_days=0
        )

        # Only subjects A and B should be in result (inner join)
        subjects_in_result = result["subject_id"].unique()
        self.assertIn("A", subjects_in_result)
        self.assertIn("B", subjects_in_result)
        self.assertNotIn("C", subjects_in_result)


if __name__ == "__main__":
    unittest.main()
