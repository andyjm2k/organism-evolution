"""Unit tests for distance helpers (squared vs linear contracts)."""

import math
import sys
import unittest
from pathlib import Path

# Allow imports from src/ when running tests from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distance import colliding, distance, squared_distance, within_radius


class TestDistance(unittest.TestCase):
    """Verify distance unit contracts used by sensing and collision."""

    def test_squared_distance_matches_manual(self):
        # Squared distance should equal dx^2 + dy^2 without a sqrt.
        self.assertEqual(squared_distance((0, 0), (3, 4)), 25)

    def test_distance_is_linear(self):
        # Linear distance should be the Euclidean length.
        self.assertAlmostEqual(distance((0, 0), (3, 4)), 5.0)

    def test_within_radius_uses_linear_radius(self):
        # A point 50 units away is inside radius 300 (the old bug missed this).
        self.assertTrue(within_radius((0, 0), (50, 0), 300))
        # But outside a too-small radius.
        self.assertFalse(within_radius((0, 0), (50, 0), 40))

    def test_within_radius_rejects_non_positive(self):
        # Zero/negative radii never contain another point usefully.
        self.assertFalse(within_radius((0, 0), (0, 0), 0))
        self.assertFalse(within_radius((0, 0), (1, 0), -5))

    def test_colliding_circles(self):
        # Circles of radius 10 centered 15 apart overlap.
        self.assertTrue(colliding((0, 0), 10, (15, 0), 10))
        # Circles of radius 5 centered 20 apart do not overlap.
        self.assertFalse(colliding((0, 0), 5, (20, 0), 5))

    def test_none_positions_are_infinitely_far(self):
        # Missing positions are treated as infinitely far.
        self.assertTrue(math.isinf(squared_distance(None, (0, 0))))
        self.assertTrue(math.isinf(distance((0, 0), None)))


if __name__ == "__main__":
    unittest.main()
