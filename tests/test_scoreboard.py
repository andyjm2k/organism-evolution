"""Unit tests for scoreboard record and dashboard behavior."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scoreboard import Scoreboard


def _organism_stub(is_carnivore=False):
    """Minimal organism stand-in for record_species."""
    return SimpleNamespace(
        get_radius=lambda: 5,
        calculate_spikes=lambda: 4,
        calculate_spike_length=lambda: 3,
        is_carnivore=is_carnivore,
    )


class TestScoreboard(unittest.TestCase):
    """Cover last_seen updates and dashboard level plumbing."""

    def setUp(self):
        Scoreboard.initialize(dashboard_level="normal")

    def test_last_seen_updates_when_fitness_not_improved(self):
        # Species should remain tracked even when fitness drops.
        org = _organism_stub()
        Scoreboard.record_species("1", org, fitness=100, generation=1, config=None)
        Scoreboard.record_species("1", org, fitness=50, generation=3, config=None)
        record = Scoreboard.get_records()["1"]
        self.assertEqual(record["last_seen"], 3)
        self.assertEqual(record["highest_fitness"], 100)

    def test_dashboard_level_is_configurable(self):
        # Dashboard level should not rely on stack introspection.
        Scoreboard.set_dashboard_level("minimal")
        self.assertEqual(Scoreboard._dashboard_level, "minimal")


if __name__ == "__main__":
    unittest.main()
