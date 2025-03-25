import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from ehrviz.stats.frequency.tree_map import plot_treemap


class TestTreeMap(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_counts = pd.Series({"CODE1": 50, "CODE2": 30, "CODE3": 20})

        self.sample_rename_dict = {
            "CODE1": "Type A",
            "CODE2": "Type B",
            "CODE3": "Type C",
        }

    def test_plot_treemap_basic_input(self):
        """Test plot_treemap with basic input."""
        with patch("matplotlib.pyplot.show") as mock_show:
            plot_treemap(self.sample_counts)
            mock_show.assert_called_once()

    def test_plot_treemap_with_rename(self):
        """Test plot_treemap with rename dictionary."""
        with patch(
            "ehrviz.stats.frequency.tree_map.prepare_treemap_data"
        ) as mock_prepare:
            # Set up mock return values
            mock_prepare.return_value = (
                np.array([50.0, 30.0, 20.0]),
                ["Type A", "Type B", "Type C"],
            )

            with patch("matplotlib.pyplot.show"):
                plot_treemap(self.sample_counts, rename_dict=self.sample_rename_dict)

            mock_prepare.assert_called_once_with(
                self.sample_counts, self.sample_rename_dict
            )

    def test_plot_treemap_save_figure(self):
        """Test plot_treemap saves figure when save_path is provided."""
        with patch("matplotlib.figure.Figure.savefig") as mock_save:
            with patch("matplotlib.pyplot.show"):
                with patch(
                    "matplotlib.pyplot.subplots",
                    return_value=(MagicMock(), MagicMock()),
                ):
                    plot_treemap(self.sample_counts, save_path="test.png")

            mock_save.assert_called_once()

    def test_plot_treemap_custom_dimensions(self):
        """Test plot_treemap with custom width and height."""
        custom_width = 200
        custom_height = 150

        with patch(
            "ehrviz.stats.frequency.tree_map.compute_treemap_rectangles"
        ) as mock_compute:
            with patch("matplotlib.pyplot.show"):
                with patch(
                    "matplotlib.pyplot.subplots",
                    return_value=(MagicMock(), MagicMock()),
                ):
                    plot_treemap(
                        self.sample_counts, width=custom_width, height=custom_height
                    )

            # Verify compute_treemap_rectangles was called with custom dimensions
            mock_compute.assert_called_once()
            args = mock_compute.call_args[0]
            self.assertEqual(args[1], custom_width)
            self.assertEqual(args[2], custom_height)

    def test_plot_treemap_custom_thresholds(self):
        """Test plot_treemap with custom annotation thresholds."""
        with patch("ehrviz.stats.frequency.tree_map.draw_treemap") as mock_draw:
            with patch("matplotlib.pyplot.show"):
                with patch(
                    "matplotlib.pyplot.subplots",
                    return_value=(MagicMock(), MagicMock()),
                ):
                    plot_treemap(
                        self.sample_counts, big_threshold=10, medium_threshold=8
                    )

            # Verify draw_treemap was called with custom thresholds
            mock_draw.assert_called_once()
            _, _, _, _, big_threshold, medium_threshold = mock_draw.call_args[0]
            self.assertEqual(big_threshold, 10)
            self.assertEqual(medium_threshold, 8)

    def test_plot_treemap_axis_configuration(self):
        """Test plot_treemap axis configuration."""
        mock_ax = MagicMock()
        mock_fig = MagicMock()

        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.show"):
                # Test with axis_off=True (default)
                plot_treemap(self.sample_counts)
                mock_ax.axis.assert_called_with("off")

                mock_ax.reset_mock()

                # Test with axis_off=False
                plot_treemap(self.sample_counts, axis_off=False)
                mock_ax.axis.assert_not_called()

    def test_plot_treemap_custom_title(self):
        """Test plot_treemap with custom title and font size."""
        custom_title = "Custom Treemap Title"
        custom_fontsize = 20

        mock_ax = MagicMock()
        mock_fig = MagicMock()

        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.show"):
                plot_treemap(
                    self.sample_counts,
                    title=custom_title,
                    title_fontsize=custom_fontsize,
                )

            mock_ax.set_title.assert_called_with(custom_title, fontsize=custom_fontsize)

    def tearDown(self):
        """Clean up after each test method."""
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
