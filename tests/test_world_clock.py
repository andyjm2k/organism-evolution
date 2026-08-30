"""Unit tests for WorldClock."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world_clock import WorldClock


class TestWorldClock(unittest.TestCase):
    """Cover step counting, epoch rollover, and max-step limits."""

    def test_tick_increments_step(self):
        clock = WorldClock(epoch_length=10)
        clock.tick()
        self.assertEqual(clock.step, 1)

    def test_epoch_increments_at_boundary(self):
        clock = WorldClock(epoch_length=5)
        for _ in range(5):
            clock.tick()
        self.assertEqual(clock.epoch, 1)
        self.assertEqual(clock.step, 5)

    def test_alive_respects_max_steps(self):
        clock = WorldClock()
        self.assertTrue(clock.alive(None))
        self.assertTrue(clock.alive(10))
        clock.step = 9
        self.assertTrue(clock.alive(10))
        clock.step = 10
        self.assertFalse(clock.alive(10))

    def test_epoch_length_minimum_one(self):
        clock = WorldClock(epoch_length=0)
        clock.tick()
        self.assertEqual(clock.epoch_length, 1)


if __name__ == "__main__":
    unittest.main()
