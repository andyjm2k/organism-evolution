"""Headless tests for shared renderer helpers (RendererCommon)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from renderer_common import RendererCommon
from scoreboard import Scoreboard


class TestRendererCommon(unittest.TestCase):
    """Cover scoreboard surface building and color helpers without a display."""

    def setUp(self):
        """Use a lightweight RendererCommon instance (no pygame window)."""
        self.common = RendererCommon(logging_level="normal")

    def test_species_color_normalized_in_unit_interval(self):
        """GPU colors should be RGB floats in 0..1."""
        red, green, blue = self.common.species_color_normalized("42", False)
        for channel in (red, green, blue):
            self.assertGreaterEqual(channel, 0.0)
            self.assertLessEqual(channel, 1.0)

    def test_build_scoreboard_surface_waiting_state(self):
        """Empty scoreboard records produce a valid waiting surface."""
        Scoreboard.initialize()
        surface = self.common.build_scoreboard_surface(300)
        self.assertEqual(surface.get_width(), self.common.scoreboard_width)
        self.assertGreater(surface.get_height(), 0)

    def test_get_species_visual_caches_surface(self):
        """Species icon surfaces should be cached by visual parameters."""
        first = self.common.get_species_visual("9", True, 6, 4, 3, num_nodes=10)
        second = self.common.get_species_visual("9", True, 6, 4, 3, num_nodes=10)
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
