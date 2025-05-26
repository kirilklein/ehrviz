import unittest

import pandas as pd

from ehrviz.survival.prepare import (
    _add_survival_and_cumulative,
    _compute_daily_counts,
    _filter_subjects,
    _initialize_weights,
    calculate_cumulative_incidence,
)


class TestFilterSubjects(unittest.TestCase):
    """Test subject filtering functionality."""

    def setUp(self):
        self.data = pd.DataFrame(
            {
                "subject_id": ["A", "A", "B", "B", "C", "C"],
                "day": [0, 1, 0, 1, 0, 2],
                "outcome": [0, 1, 0, 0, 0, 0],
                "censored": [0, 0, 0, 1, 0, 1],
            }
        )

    def test_no_filter_returns_copy(self):
        """Test that no filter returns a copy of the data."""
        result = _filter_subjects(self.data, subject_list=None)

        # Should be equal but not the same object
        pd.testing.assert_frame_equal(result, self.data)
        self.assertIsNot(result, self.data)

    def test_filter_specific_subjects(self):
        """Test filtering for specific subjects."""
        result = _filter_subjects(self.data, subject_list=["A", "C"])

        expected_subjects = ["A", "C"]
        actual_subjects = result["subject_id"].unique().tolist()

        self.assertEqual(sorted(actual_subjects), sorted(expected_subjects))
        self.assertEqual(len(result), 4)  # 2 records for A, 2 for C

    def test_filter_nonexistent_subjects(self):
        """Test filtering for subjects not in data."""
        result = _filter_subjects(self.data, subject_list=["X", "Y"])

        self.assertEqual(len(result), 0)

    def test_filter_empty_list(self):
        """Test filtering with empty subject list."""
        result = _filter_subjects(self.data, subject_list=[])

        self.assertEqual(len(result), 0)


class TestInitializeWeights(unittest.TestCase):
    """Test weight initialization functionality."""

    def setUp(self):
        self.data = pd.DataFrame(
            {
                "subject_id": ["A", "A", "B", "B", "C"],
                "day": [0, 1, 0, 1, 0],
                "outcome": [0, 1, 0, 0, 0],
                "censored": [0, 0, 0, 1, 1],
            }
        )

    def test_no_weights_returns_default(self):
        """Test that no weights returns default weights of 1.0."""
        result = _initialize_weights(self.data, weights=None)

        expected = {"A": 1.0, "B": 1.0, "C": 1.0}
        self.assertEqual(result, expected)

    def test_provided_weights_returned(self):
        """Test that provided weights are returned as-is."""
        custom_weights = {"A": 2.0, "B": 0.5, "C": 1.5}
        result = _initialize_weights(self.data, weights=custom_weights)

        self.assertEqual(result, custom_weights)

    def test_partial_weights_returned(self):
        """Test that partial weights are returned (missing subjects handled elsewhere)."""
        partial_weights = {"A": 2.0, "B": 0.5}  # Missing C
        result = _initialize_weights(self.data, weights=partial_weights)

        self.assertEqual(result, partial_weights)


class TestComputeDailyCounts(unittest.TestCase):
    """Test daily count computation with subjects removed from at-risk on event day."""

    def setUp(self):
        # Test case: 3 subjects with outcomes and censoring
        self.data = pd.DataFrame(
            {
                "subject_id": ["A", "A", "A", "B", "B", "C", "C", "C"],
                "day": [0, 1, 2, 0, 1, 0, 1, 2],
                "outcome": [
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                ],  # A outcome on day 2, B outcome on day 1
                "censored": [0, 0, 0, 0, 0, 0, 0, 1],  # C censored on day 2
            }
        )
        self.weights = {"A": 1.0, "B": 2.0, "C": 0.5}

    def test_basic_daily_counts_with_event_removal(self):
        """Test daily count computation with subjects removed on event day."""
        result = _compute_daily_counts(self.data, self.weights)

        # Check structure
        expected_columns = ["day", "at_risk", "events", "subjects_at_risk"]
        self.assertListEqual(list(result.columns), expected_columns)
        self.assertEqual(len(result), 3)  # Days 0, 1, 2

        # Day 0: All subjects at risk (A=1.0, B=2.0, C=0.5), no events
        self.assertEqual(result.loc[0, "at_risk"], 3.5)
        self.assertEqual(result.loc[0, "events"], 0.0)
        self.assertEqual(result.loc[0, "subjects_at_risk"], 3)

        # Day 1: All subjects would be at risk, but B has outcome so removed
        # At risk: A=1.0, C=0.5 (B removed due to outcome)
        self.assertEqual(result.loc[1, "at_risk"], 1.5)
        self.assertEqual(result.loc[1, "events"], 2.0)  # B outcome (weight=2.0)
        self.assertEqual(result.loc[1, "subjects_at_risk"], 2)  # A, C

        # Day 2: A and C would be at risk, but both have events (A outcome, C censored)
        # At risk: none (both A and C removed)
        self.assertEqual(result.loc[2, "at_risk"], 0.0)
        self.assertEqual(result.loc[2, "events"], 1.0)  # A outcome (weight=1.0)
        self.assertEqual(result.loc[2, "subjects_at_risk"], 0)

    def test_censoring_removes_from_at_risk(self):
        """Test that censoring on a day removes subject from at-risk for that day."""
        data_with_censoring = pd.DataFrame(
            {
                "subject_id": ["A", "A", "B", "B"],
                "day": [0, 1, 0, 1],
                "outcome": [0, 0, 0, 0],
                "censored": [0, 1, 0, 0],  # A censored on day 1
            }
        )
        weights = {"A": 1.0, "B": 1.0}

        result = _compute_daily_counts(data_with_censoring, weights)

        # Day 0: Both at risk
        self.assertEqual(result.loc[0, "at_risk"], 2.0)
        self.assertEqual(result.loc[0, "subjects_at_risk"], 2)

        # Day 1: A is censored, so removed from at-risk
        self.assertEqual(result.loc[1, "at_risk"], 1.0)  # Only B
        self.assertEqual(result.loc[1, "subjects_at_risk"], 1)  # Only B

    def test_outcome_and_censoring_same_day(self):
        """Test when outcome and censoring happen on same day for different subjects."""
        data_mixed = pd.DataFrame(
            {
                "subject_id": ["A", "A", "B", "B"],
                "day": [0, 1, 0, 1],
                "outcome": [0, 1, 0, 0],  # A has outcome on day 1
                "censored": [0, 0, 0, 1],  # B censored on day 1
            }
        )
        weights = {"A": 1.0, "B": 1.0}

        result = _compute_daily_counts(data_mixed, weights)

        # Day 0: Both at risk
        self.assertEqual(result.loc[0, "at_risk"], 2.0)

        # Day 1: Both subjects have events, so both removed from at-risk
        self.assertEqual(result.loc[1, "at_risk"], 0.0)  # Both removed
        self.assertEqual(result.loc[1, "events"], 1.0)  # A outcome
        self.assertEqual(result.loc[1, "subjects_at_risk"], 0)

    def test_no_events_with_censoring(self):
        """Test with no events but censoring."""
        data_no_events = pd.DataFrame(
            {
                "subject_id": ["A", "A", "B", "B", "C"],
                "day": [0, 1, 0, 2, 0],
                "outcome": [0, 0, 0, 0, 0],
                "censored": [
                    0,
                    1,
                    0,
                    1,
                    1,
                ],  # A censored day 1, B censored day 2, C censored day 0
            }
        )
        weights = {"A": 1.0, "B": 1.0, "C": 1.0}

        result = _compute_daily_counts(data_no_events, weights)

        # All events should be 0
        self.assertTrue((result["events"] == 0).all())

        # Day 0: C is censored, so only A and B at risk
        self.assertEqual(result.loc[0, "at_risk"], 2.0)  # A, B (C censored)

        # Day 1: A is censored, so only B at risk
        self.assertEqual(result.loc[1, "at_risk"], 1.0)  # B (A censored)

        # Day 2: B is censored, so no one at risk
        self.assertEqual(result.loc[2, "at_risk"], 0.0)  # None (B censored)

    def test_single_day_data(self):
        """Test with data spanning only one day."""
        single_day_data = pd.DataFrame(
            {
                "subject_id": ["A", "B"],
                "day": [0, 0],
                "outcome": [1, 0],
                "censored": [0, 1],
            }
        )
        weights = {"A": 1.0, "B": 1.0}

        result = _compute_daily_counts(single_day_data, weights)

        self.assertEqual(len(result), 1)  # Only day 0
        # Both subjects have events on day 0, so removed from at-risk
        self.assertEqual(result.loc[0, "at_risk"], 0.0)
        self.assertEqual(result.loc[0, "events"], 1.0)  # A outcome
        self.assertEqual(result.loc[0, "subjects_at_risk"], 0)

    def test_multiple_subjects_same_outcome_day(self):
        """Test multiple subjects with outcomes on the same day."""
        data_multiple_outcomes = pd.DataFrame(
            {
                "subject_id": ["A", "A", "B", "B", "C", "C"],
                "day": [0, 1, 0, 1, 0, 1],
                "outcome": [0, 1, 0, 1, 0, 0],  # A and B both have outcomes on day 1
                "censored": [0, 0, 0, 0, 0, 1],  # C censored on day 1
            }
        )
        weights = {"A": 1.0, "B": 2.0, "C": 0.5}

        result = _compute_daily_counts(data_multiple_outcomes, weights)

        # Day 0: All at risk
        self.assertEqual(result.loc[0, "at_risk"], 3.5)  # A + B + C

        # Day 1: All subjects have events (A outcome, B outcome, C censored), so none at risk
        self.assertEqual(result.loc[1, "at_risk"], 0.0)
        self.assertEqual(result.loc[1, "events"], 3.0)  # A(1.0) + B(2.0) outcomes
        self.assertEqual(result.loc[1, "subjects_at_risk"], 0)


class TestAddSurvivalAndCumulative(unittest.TestCase):
    """Test survival and cumulative incidence calculations."""

    def test_basic_calculations(self):
        """Test basic survival calculations."""
        daily_counts = pd.DataFrame(
            {
                "day": [0, 1, 2],
                "at_risk": [100.0, 80.0, 60.0],
                "events": [0.0, 20.0, 10.0],
                "subjects_at_risk": [100, 80, 60],
            }
        )

        result = _add_survival_and_cumulative(daily_counts)

        # Check new columns added
        expected_columns = [
            "day",
            "at_risk",
            "events",
            "subjects_at_risk",
            "hazard",
            "survival",
            "cumulative_incidence",
        ]
        self.assertListEqual(list(result.columns), expected_columns)

        # Check calculations
        # Day 0: hazard=0/100=0, survival=1, cum_inc=0
        self.assertEqual(result.loc[0, "hazard"], 0.0)
        self.assertEqual(result.loc[0, "survival"], 1.0)
        self.assertEqual(result.loc[0, "cumulative_incidence"], 0.0)

        # Day 1: hazard=20/80=0.25, survival=1*0.75=0.75, cum_inc=25%
        self.assertAlmostEqual(result.loc[1, "hazard"], 0.25)
        self.assertAlmostEqual(result.loc[1, "survival"], 0.75)
        self.assertAlmostEqual(result.loc[1, "cumulative_incidence"], 25.0)

        # Day 2: hazard=10/60≈0.167, survival=0.75*0.833≈0.625, cum_inc≈37.5%
        self.assertAlmostEqual(result.loc[2, "hazard"], 10 / 60, places=3)
        self.assertAlmostEqual(
            result.loc[2, "survival"], 0.75 * (1 - 10 / 60), places=3
        )
        self.assertAlmostEqual(
            result.loc[2, "cumulative_incidence"],
            (1 - 0.75 * (1 - 10 / 60)) * 100,
            places=1,
        )

    def test_zero_at_risk(self):
        """Test handling of zero at-risk subjects."""
        daily_counts = pd.DataFrame(
            {
                "day": [0, 1, 2],
                "at_risk": [10.0, 0.0, 0.0],  # Zero at risk on days 1, 2
                "events": [5.0, 2.0, 1.0],
                "subjects_at_risk": [10, 0, 0],
            }
        )

        result = _add_survival_and_cumulative(daily_counts)

        # Day 0: normal calculation
        self.assertEqual(result.loc[0, "hazard"], 0.5)

        # Days 1, 2: hazard should be 0 when at_risk is 0
        self.assertEqual(result.loc[1, "hazard"], 0.0)
        self.assertEqual(result.loc[2, "hazard"], 0.0)

        # Survival should remain constant after day 0
        expected_survival = 0.5  # 1 - 0.5
        self.assertEqual(result.loc[1, "survival"], expected_survival)
        self.assertEqual(result.loc[2, "survival"], expected_survival)


class TestCalculateCumulativeIncidenceIntegration(unittest.TestCase):
    """Integration tests for the main function with corrected at-risk logic."""

    def setUp(self):
        """Set up test survival data."""
        self.survival_data = pd.DataFrame(
            {
                "subject_id": ["A", "A", "A", "B", "B", "C", "C", "C", "D", "D"],
                "day": [0, 1, 2, 0, 1, 0, 1, 2, 0, 3],
                "outcome": [
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],  # A outcome day 2, B outcome day 1
                "censored": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                ],  # C censored day 2, D censored day 3
            }
        )

    def test_basic_functionality_with_corrected_logic(self):
        """Test basic cumulative incidence calculation with corrected at-risk logic."""
        result = calculate_cumulative_incidence(self.survival_data)

        # Check structure
        expected_columns = [
            "day",
            "at_risk",
            "events",
            "subjects_at_risk",
            "hazard",
            "survival",
            "cumulative_incidence",
        ]
        self.assertListEqual(list(result.columns), expected_columns)

        # Should have days 0 through 3
        self.assertEqual(len(result), 4)
        self.assertListEqual(result["day"].tolist(), [0, 1, 2, 3])

        # Day 0: All 4 subjects at risk, no events
        self.assertEqual(result.loc[0, "at_risk"], 4.0)
        self.assertEqual(result.loc[0, "events"], 0.0)
        self.assertEqual(result.loc[0, "cumulative_incidence"], 0.0)

        # Day 1: B has outcome, so removed from at-risk
        # At risk: A, C, D (B removed due to outcome)
        self.assertEqual(result.loc[1, "at_risk"], 3.0)
        self.assertEqual(result.loc[1, "events"], 1.0)

        # Day 2: A has outcome, C is censored, so both removed from at-risk
        # At risk: D only
        self.assertEqual(result.loc[2, "at_risk"], 1.0)  # Only D
        self.assertEqual(result.loc[2, "events"], 1.0)  # A outcome

        # Day 3: D is censored, so removed from at-risk
        self.assertEqual(result.loc[3, "at_risk"], 0.0)  # D censored
        self.assertEqual(result.loc[3, "events"], 0.0)

        # Cumulative incidence should be non-decreasing
        cum_inc = result["cumulative_incidence"].tolist()
        self.assertTrue(
            all(cum_inc[i] <= cum_inc[i + 1] for i in range(len(cum_inc) - 1))
        )

    def test_weighted_analysis_with_corrected_logic(self):
        """Test weighted cumulative incidence calculation with corrected logic."""
        weights = {"A": 2.0, "B": 0.5, "C": 1.0, "D": 1.5}

        result = calculate_cumulative_incidence(self.survival_data, weights=weights)

        # Day 0: weighted at risk = 2.0 + 0.5 + 1.0 + 1.5 = 5.0
        self.assertEqual(result.loc[0, "at_risk"], 5.0)

        # Day 1: B has outcome (weight 0.5), removed from at-risk
        # At risk: A(2.0) + C(1.0) + D(1.5) = 4.5
        self.assertEqual(result.loc[1, "at_risk"], 4.5)
        self.assertEqual(result.loc[1, "events"], 0.5)

        # Day 2: A outcome (weight 2.0), C censored (weight 1.0), both removed
        # At risk: D(1.5)
        self.assertEqual(result.loc[2, "at_risk"], 1.5)
        self.assertEqual(result.loc[2, "events"], 2.0)  # A outcome

        # Day 3: D censored, removed from at-risk
        self.assertEqual(result.loc[3, "at_risk"], 0.0)

    def test_all_censored_same_day(self):
        """Test when all remaining subjects are censored on the same day."""
        censored_data = pd.DataFrame(
            {
                "subject_id": ["A", "A", "B", "B"],
                "day": [0, 1, 0, 1],
                "outcome": [0, 0, 0, 0],
                "censored": [0, 1, 0, 1],  # Both censored on day 1
            }
        )

        result = calculate_cumulative_incidence(censored_data)

        # Day 0: Both at risk
        self.assertEqual(result.loc[0, "at_risk"], 2.0)

        # Day 1: Both censored, so removed from at-risk
        self.assertEqual(result.loc[1, "at_risk"], 0.0)
        self.assertEqual(result.loc[1, "events"], 0.0)

        # Cumulative incidence should remain 0
        self.assertTrue((result["cumulative_incidence"] == 0).all())

    def test_empty_data(self):
        """Test with empty survival data."""
        empty_data = pd.DataFrame(columns=["subject_id", "day", "outcome", "censored"])

        result = calculate_cumulative_incidence(empty_data)

        # Should return empty DataFrame with correct columns
        expected_columns = [
            "day",
            "at_risk",
            "events",
            "subjects_at_risk",
            "hazard",
            "survival",
            "cumulative_incidence",
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        self.assertEqual(len(result), 0)

    def test_kaplan_meier_method_explicit(self):
        """Test Kaplan-Meier method explicitly specified."""
        result = calculate_cumulative_incidence(self.survival_data, method="kaplan_meier")
        
        # Should be same as default behavior
        result_default = calculate_cumulative_incidence(self.survival_data)
        pd.testing.assert_frame_equal(result, result_default)

    def test_crude_method(self):
        """Test crude cumulative incidence method."""
        result = calculate_cumulative_incidence(self.survival_data, method="crude")
        
        # Check structure
        expected_columns = [
            "day",
            "at_risk",
            "events",
            "subjects_at_risk",
            "hazard",
            "survival",
            "cumulative_incidence",
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        
        # Initial at-risk should be 4
        initial_at_risk = 4.0
        
        # Day 0: No events, cumulative incidence = 0
        self.assertEqual(result.loc[0, "cumulative_incidence"], 0.0)
        
        # Day 1: 1 event out of initial 4, crude hazard = 1/4 = 25%
        expected_hazard_day1 = 1.0 / initial_at_risk
        self.assertAlmostEqual(result.loc[1, "hazard"], expected_hazard_day1)
        self.assertAlmostEqual(result.loc[1, "cumulative_incidence"], 25.0)
        
        # Day 2: 1 more event, crude hazard = 1/4 = 25%, cumulative = 50%
        expected_hazard_day2 = 1.0 / initial_at_risk
        self.assertAlmostEqual(result.loc[2, "hazard"], expected_hazard_day2)
        self.assertAlmostEqual(result.loc[2, "cumulative_incidence"], 50.0)
        
        # Day 3: No events, cumulative incidence stays at 50%
        self.assertAlmostEqual(result.loc[3, "hazard"], 0.0)
        self.assertAlmostEqual(result.loc[3, "cumulative_incidence"], 50.0)

    def test_crude_vs_kaplan_meier_comparison(self):
        """Test that crude and Kaplan-Meier methods give different results."""
        result_km = calculate_cumulative_incidence(self.survival_data, method="kaplan_meier")
        result_crude = calculate_cumulative_incidence(self.survival_data, method="crude")
        
        # They should have the same structure
        self.assertListEqual(list(result_km.columns), list(result_crude.columns))
        self.assertEqual(len(result_km), len(result_crude))
        
        # But different cumulative incidence values (except at day 0)
        self.assertEqual(result_km.loc[0, "cumulative_incidence"], 
                        result_crude.loc[0, "cumulative_incidence"])  # Both 0 at start
        
        # After events occur, they should differ
        self.assertNotEqual(result_km.loc[2, "cumulative_incidence"], 
                           result_crude.loc[2, "cumulative_incidence"])

    def test_crude_method_with_weights(self):
        """Test crude method with weights."""
        weights = {"A": 2.0, "B": 0.5, "C": 1.0, "D": 1.5}
        result = calculate_cumulative_incidence(self.survival_data, weights=weights, method="crude")
        
        # Initial weighted at-risk = 2.0 + 0.5 + 1.0 + 1.5 = 5.0
        initial_at_risk = 5.0
        
        # Day 1: B outcome (weight 0.5), crude hazard = 0.5/5.0 = 10%
        expected_hazard_day1 = 0.5 / initial_at_risk
        self.assertAlmostEqual(result.loc[1, "hazard"], expected_hazard_day1)
        self.assertAlmostEqual(result.loc[1, "cumulative_incidence"], 10.0)
        
        # Day 2: A outcome (weight 2.0), crude hazard = 2.0/5.0 = 40%
        # Cumulative = 10% + 40% = 50%
        expected_hazard_day2 = 2.0 / initial_at_risk
        self.assertAlmostEqual(result.loc[2, "hazard"], expected_hazard_day2)
        self.assertAlmostEqual(result.loc[2, "cumulative_incidence"], 50.0)

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_cumulative_incidence(self.survival_data, method="invalid_method")
        
        self.assertIn("method must be either 'kaplan_meier' or 'crude'", str(context.exception))

    def test_crude_method_single_subject(self):
        """Test crude method with single subject."""
        single_subject_data = pd.DataFrame({
            'subject_id': ['A', 'A', 'A'],
            'day': [0, 1, 2],
            'outcome': [0, 0, 1],
            'censored': [0, 0, 0]
        })
        
        result = calculate_cumulative_incidence(single_subject_data, method="crude")
        
        # Initial at-risk = 1, outcome on day 2
        # Day 2: hazard = 1/1 = 100%, cumulative incidence = 100%
        self.assertEqual(result.loc[2, "cumulative_incidence"], 100.0)
        self.assertEqual(result.loc[2, "survival"], 0.0)

    def test_crude_method_no_events(self):
        """Test crude method with no events (all censored)."""
        censored_data = pd.DataFrame({
            'subject_id': ['A', 'A', 'B', 'B'],
            'day': [0, 1, 0, 1],
            'outcome': [0, 0, 0, 0],
            'censored': [0, 1, 0, 1]
        })
        
        result = calculate_cumulative_incidence(censored_data, method="crude")
        
        # No events, so cumulative incidence should remain 0
        self.assertTrue((result['cumulative_incidence'] == 0).all())
        self.assertTrue((result['survival'] == 1.0).all())


# Add these test cases to test the helper functions
class TestCrudeVsKaplanMeierMethods(unittest.TestCase):
    """Test the different cumulative incidence calculation methods."""
    
    def test_kaplan_meier_helper(self):
        """Test Kaplan-Meier helper function."""
        daily_counts = pd.DataFrame({
            'day': [0, 1, 2],
            'at_risk': [100.0, 80.0, 60.0],
            'events': [0.0, 20.0, 10.0],
            'subjects_at_risk': [100, 80, 60]
        })
        
        result = _add_survival_and_cumulative_km(daily_counts)
        
        # Day 1: hazard = 20/80 = 0.25, survival = 0.75, cum_inc = 25%
        self.assertAlmostEqual(result.loc[1, 'hazard'], 0.25)
        self.assertAlmostEqual(result.loc[1, 'cumulative_incidence'], 25.0)
    
    def test_crude_helper(self):
        """Test crude cumulative incidence helper function."""
        daily_counts = pd.DataFrame({
            'day': [0, 1, 2],
            'at_risk': [100.0, 80.0, 60.0],  # at_risk shrinks but we use initial
            'events': [0.0, 20.0, 10.0],
            'subjects_at_risk': [100, 80, 60]
        })
        
        result = _add_survival_and_cumulative_crude(daily_counts)
        
        # Day 1: crude hazard = 20/100 = 0.20, cum_inc = 20%
        self.assertAlmostEqual(result.loc[1, 'hazard'], 0.20)
        self.assertAlmostEqual(result.loc[1, 'cumulative_incidence'], 20.0)
        
        # Day 2: crude hazard = 10/100 = 0.10, cum_inc = 20% + 10% = 30%
        self.assertAlmostEqual(result.loc[2, 'hazard'], 0.10)
        self.assertAlmostEqual(result.loc[2, 'cumulative_incidence'], 30.0)
    
    def test_empty_dataframe_crude(self):
        """Test crude method with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['day', 'at_risk', 'events', 'subjects_at_risk'])
        result = _add_survival_and_cumulative_crude(empty_df)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
