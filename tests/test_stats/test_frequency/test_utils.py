import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ehrviz.stats.frequency.utils import (
    compute_treemap_rectangles,
    filter_counts_by_pattern,
    group_counts,
    prepare_treemap_data,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Sample data for testing
        self.sample_counts = pd.Series({"A1": 10, "A2": 15, "B1": 5})
        self.sample_rename_dict = {"A1": "A", "A2": "A", "B1": "B"}

    def test_group_counts_basic(self):
        """Test basic functionality of group_counts."""
        result = group_counts(self.sample_counts, self.sample_rename_dict)

        expected = pd.Series({"A": 25, "B": 5})

        pd.testing.assert_series_equal(result, expected)

    def test_group_counts_missing_mapping(self):
        """Test group_counts with missing mappings."""
        rename_dict = {"A1": "A"}  # Deliberately missing some mappings

        result = group_counts(self.sample_counts, rename_dict)

        expected = pd.Series({"A": 10, "A2": 15, "B1": 5}).sort_values(ascending=False)

        pd.testing.assert_series_equal(result, expected)

    def test_prepare_treemap_data_series_input(self):
        """Test prepare_treemap_data with Series input."""
        sizes, codes = prepare_treemap_data(self.sample_counts)

        expected_sizes = np.array([1000.0, 1500.0, 500.0])  # Values * 100
        expected_codes = ["A1", "A2", "B1"]

        np.testing.assert_array_almost_equal(sizes, expected_sizes)
        self.assertEqual(codes, expected_codes)

    def test_prepare_treemap_data_with_rename(self):
        """Test prepare_treemap_data with rename dictionary."""
        sizes, codes = prepare_treemap_data(
            self.sample_counts, rename_dict=self.sample_rename_dict
        )

        expected_sizes = np.array([1000.0, 1500.0, 500.0])
        expected_codes = ["A", "A", "B"]

        np.testing.assert_array_almost_equal(sizes, expected_sizes)
        self.assertEqual(codes, expected_codes)

    def test_compute_treemap_rectangles(self):
        """Test compute_treemap_rectangles basic functionality."""
        sizes = np.array([60.0, 30.0, 10.0])
        width = 100
        height = 100

        rects = compute_treemap_rectangles(sizes, width, height)

        # Basic checks on the output
        self.assertEqual(len(rects), 3)
        for rect in rects:
            self.assertTrue(all(key in rect for key in ["x", "y", "dx", "dy"]))
            self.assertTrue(0 <= rect["x"] <= width)
            self.assertTrue(0 <= rect["y"] <= height)
            self.assertTrue(0 <= rect["dx"] <= width)
            self.assertTrue(0 <= rect["dy"] <= height)

    def test_filter_counts_by_pattern_basic(self):
        """Test basic regex pattern filtering."""
        counts = pd.Series({"L/A1": 10, "L/A2": 5, "D/B1": 20, "M/C1": 15})

        # Test filtering lab codes
        result = filter_counts_by_pattern(counts, "^L/")
        expected = pd.Series({"L/A1": 10, "L/A2": 5})
        pd.testing.assert_series_equal(result, expected)

    def tearDown(self):
        """Clean up after each test."""
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
