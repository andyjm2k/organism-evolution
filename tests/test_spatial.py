"""Unit tests for the spatial grid proximity index."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial import SpatialGrid


class TestSpatialGrid(unittest.TestCase):
    """Verify grid queries match linear radius semantics."""

    def test_query_finds_nearby_entity(self):
        # Entity 40 units away should be found within radius 100.
        grid = SpatialGrid(cell_size=50)
        near = SimpleNamespace(position=(40, 0))
        far = SimpleNamespace(position=(500, 0))
        grid.insert(near, near.position)
        grid.insert(far, far.position)
        hits = list(grid.query((0, 0), 100))
        self.assertIn(near, hits)
        self.assertNotIn(far, hits)

    def test_query_skips_none_positions(self):
        # Entities without positions must not be returned.
        grid = SpatialGrid(cell_size=50)
        missing = SimpleNamespace(position=None)
        grid.insert(missing, missing.position)
        self.assertEqual(list(grid.query((0, 0), 100)), [])


if __name__ == "__main__":
    unittest.main()
